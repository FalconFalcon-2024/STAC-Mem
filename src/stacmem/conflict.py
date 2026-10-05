"""Online, local-neighborhood conflict detection and version materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from .admission import FactualityAdmissionGate, detect_nonfactual_signals
from .models import (
    AdmissionDecision,
    Claim,
    ClaimDraft,
    ClaimStatus,
    ConflictRelation,
    RelationType,
    UpdateKind,
    normalize_text,
)
from .spatial import places_disjoint
from .store import ClaimStore
from .temporal_eligibility import evidence_interval, evidence_overlaps
from .temporal_grounding import TemporalGrounder
from .time_utils import overlaps


@dataclass(frozen=True)
class PredicatePolicy:
    functional: bool
    stateful: bool = True
    spatially_scoped: bool = False


class PredicatePolicyRegistry:
    """Small explicit policy table with conservative defaults."""

    _nonfunctional_markers = (
        "likes",
        "hobby",
        "visited",
        "owns",
        "friend",
        "skill",
        "activity",
    )
    _event_markers = ("attended", "visited", "purchased", "completed", "event")
    _aliases: ClassVar[dict[str, str]] = {
        "current_residence": "residence.current",
        "home.city": "residence.current",
        "lives_in": "residence.current",
        "employment.current": "employment.organization",
        "job.employer": "employment.organization",
        "employer.current": "employment.organization",
        "works_at": "employment.organization",
        "employment.previous": "employment.organization.previous",
        "occupation.current": "employment.position",
        "job.position": "employment.position",
        "job.title": "employment.position",
        "position.current": "employment.position",
        "spoken_language": "language.spoken",
        "travel_plan": "plan.travel",
        "trip.planned": "plan.travel",
        "travel.itinerary": "plan.travel",
        "item.location.current": "item.location",
        "appointment.time.current": "appointment.time",
    }

    def canonicalize(self, predicate: str) -> str:
        normalized = predicate.casefold().strip().replace(" ", "_")
        return self._aliases.get(normalized, normalized)

    def resolve(self, predicate: str, explicit: bool | None) -> PredicatePolicy:
        normalized = self.canonicalize(predicate)
        spatially_scoped = any(
            marker in normalized for marker in ("preference", "availability", "policy")
        )
        if explicit is not None:
            return PredicatePolicy(
                functional=explicit,
                stateful=True,
                spatially_scoped=spatially_scoped,
            )
        if any(marker in normalized for marker in self._event_markers):
            return PredicatePolicy(functional=False, stateful=False)
        functional = not any(marker in normalized for marker in self._nonfunctional_markers)
        return PredicatePolicy(
            functional=functional,
            stateful=functional,
            spatially_scoped=spatially_scoped,
        )


@dataclass
class ConflictOutcome:
    claim: Claim
    relations: list[ConflictRelation] = field(default_factory=list)
    mutations: list[dict] = field(default_factory=list)
    admission_decision: AdmissionDecision = AdmissionDecision.ACCEPT
    admission_reasons: list[str] = field(default_factory=list)


class ConflictEngine:
    _transition_cues: ClassVar[tuple[str, ...]] = (
        "moved to",
        "relocated to",
        "joined ",
        "left ",
        "switched to",
        "changed to",
        "started working at",
        "promoted to",
        "became ",
        "now work",
        "currently work",
        "搬到",
        "迁居",
        "加入",
        "离开",
        "转到",
        "换成",
        "改为",
        "成为",
        "现在",
        "目前",
    )
    def __init__(
        self,
        store: ClaimStore,
        *,
        close_corrected_transaction: bool = True,
        infer_transition_end: bool = True,
        infer_transition_from_evidence: bool = True,
        transition_mode: str = "lexical_v1",
        temporal_relation_mode: str = "transaction_fallback_v1",
        temporal_grounding_mode: str = "off",
        temporal_grounder: TemporalGrounder | None = None,
        admission_gate: FactualityAdmissionGate | None = None,
        policies: PredicatePolicyRegistry | None = None,
    ) -> None:
        self.store = store
        self.close_corrected_transaction = close_corrected_transaction
        self.infer_transition_end = infer_transition_end
        self.infer_transition_from_evidence = infer_transition_from_evidence
        self.transition_mode = transition_mode
        if temporal_relation_mode not in {
            "transaction_fallback_v1",
            "valid_interval_v2",
        }:
            raise ValueError(
                f"unsupported temporal relation mode: {temporal_relation_mode}"
            )
        self.temporal_relation_mode = temporal_relation_mode
        self.temporal_grounding_mode = temporal_grounding_mode
        self.temporal_grounder = temporal_grounder or TemporalGrounder()
        self.admission_gate = admission_gate or FactualityAdmissionGate()
        self.policies = policies or PredicatePolicyRegistry()

    def ingest(self, draft: ClaimDraft, *, vector: list[float] | None = None) -> ConflictOutcome:
        original_predicate = draft.predicate
        draft.predicate = self.policies.canonicalize(draft.predicate)
        if draft.predicate != original_predicate:
            draft.metadata = {
                **draft.metadata,
                "original_predicate": original_predicate,
                "predicate_normalizer": "policy-alias-v1",
            }
        if self.temporal_grounding_mode in {"certificate_v1", "certificate_v2"}:
            self.temporal_grounder.ground_and_apply(
                draft,
                mode=self.temporal_grounding_mode,
            )
        elif self.temporal_grounding_mode != "off":
            raise ValueError(
                f"unsupported temporal grounding mode: {self.temporal_grounding_mode}"
            )
        policy = self.policies.resolve(draft.predicate, draft.functional)
        draft.functional = policy.functional
        slot_history = self.store.slot_claims(draft.owner_id, draft.slot_key)
        assessment = self.admission_gate.assess(draft, existing=slot_history)
        if assessment.decision == AdmissionDecision.QUARANTINE:
            draft.metadata = {
                **draft.metadata,
                "admission_decision": assessment.decision.value,
                "admission_detector": assessment.detector,
                "admission_reasons": list(assessment.reasons),
            }
            claim = Claim.from_draft(
                draft,
                transaction_start=draft.observed_at,
                version=max((item.version for item in slot_history), default=0) + 1,
                vector=vector,
            )
            claim.status = ClaimStatus.QUARANTINED
            outcome = ConflictOutcome(
                claim=claim,
                admission_decision=assessment.decision,
                admission_reasons=list(assessment.reasons),
            )
            self.store.apply_ingest(claim, [], [])
            return outcome

        existing = [
            item for item in slot_history if item.status != ClaimStatus.QUARANTINED
        ]
        self._normalize_update_kind(draft, existing, policy)
        claim = Claim.from_draft(
            draft,
            transaction_start=draft.observed_at,
            version=max((item.version for item in slot_history), default=0) + 1,
            vector=vector,
        )
        outcome = ConflictOutcome(claim=claim, admission_decision=assessment.decision)

        for old in existing:
            relation, reason = self._classify(old, claim, policy)
            outcome.relations.append(
                ConflictRelation(
                    source_claim_id=claim.id,
                    target_claim_id=old.id,
                    relation=relation,
                    confidence=1.0,
                    reason=reason,
                    detector="structural-v1",
                )
            )
            self._materialize(old, claim, relation, outcome)

        if any(edge.relation == RelationType.CONTRADICTS for edge in outcome.relations):
            claim.status = ClaimStatus.DISPUTED
        if draft.update_kind == UpdateKind.RETRACTION:
            # The retraction is retained as an auditable event but is not a
            # positive value in the current materialized view.
            claim.status = ClaimStatus.RETRACTED

        self.store.apply_ingest(claim, outcome.relations, outcome.mutations)
        return outcome

    def _normalize_update_kind(
        self,
        draft: ClaimDraft,
        existing: list[Claim],
        policy: PredicatePolicy,
    ) -> None:
        """Upgrade a weak assertion only when independent transition evidence agrees."""
        if self.transition_mode == "proposition_v2" and draft.update_kind == UpdateKind.TRANSITION:
            certificate = draft.metadata.get("proposition_certificate", {})
            if not certificate.get("transition_entailment"):
                draft.update_kind = UpdateKind.ASSERTION
                draft.metadata = {
                    **draft.metadata,
                    "original_update_kind": UpdateKind.TRANSITION.value,
                    "update_kind_normalizer": "unsupported-transition-downgrade-v1",
                }
        if (
            not self.infer_transition_from_evidence
            or draft.update_kind != UpdateKind.ASSERTION
            or not policy.functional
            or not policy.stateful
            or not existing
        ):
            return
        new_value = normalize_text(draft.object_value)
        different = [old for old in existing if old.object_norm != new_value]
        if not different:
            return
        if self.transition_mode == "proposition_v2":
            certificate = draft.metadata.get("proposition_certificate")
            if not isinstance(certificate, dict):
                certificate = self.admission_gate.grounder.ground(draft, existing).model_dump()
                draft.metadata = {
                    **draft.metadata,
                    "proposition_certificate": certificate,
                }
            if not certificate.get("transition_entailment"):
                return
            draft.update_kind = UpdateKind.TRANSITION
            draft.metadata = {
                **draft.metadata,
                "original_update_kind": UpdateKind.ASSERTION.value,
                "update_kind_normalizer": "proposition-transition-v2",
                "transition_evidence": list(certificate.get("transition_reasons", [])),
                "transition_confidence": certificate.get("transition_confidence"),
            }
            return
        if self.transition_mode != "lexical_v1":
            raise ValueError(f"unsupported transition mode: {self.transition_mode}")
        source = normalize_text(draft.source_content)
        if normalize_text(draft.object_value) not in source:
            return
        cues = [cue for cue in self._transition_cues if cue in source]
        newer = any(
            _draft_is_newer(old, draft, mode=self.temporal_relation_mode)
            for old in different
        )
        blocked = bool(detect_nonfactual_signals(draft))
        if not cues or not newer or blocked:
            return
        draft.update_kind = UpdateKind.TRANSITION
        draft.metadata = {
            **draft.metadata,
            "original_update_kind": UpdateKind.ASSERTION.value,
            "update_kind_normalizer": "transition-evidence-v1",
            "transition_evidence": [
                "later_state_time",
                *[f"lexical:{cue.strip()}" for cue in cues],
            ],
        }

    def _classify(
        self, old: Claim, new: Claim, policy: PredicatePolicy
    ) -> tuple[RelationType, str]:
        same_value = old.object_norm == new.object_norm
        if new.update_kind == UpdateKind.RETRACTION and same_value:
            return RelationType.RETRACTS, "explicit retraction of the same normalized value"
        if same_value:
            return RelationType.SUPPORTS, "same slot and normalized value"
        if policy.spatially_scoped and places_disjoint(old.place, new.place):
            return RelationType.COEXISTS, "different non-overlapping place scopes"
        interval_overlap = (
            evidence_overlaps(old, new) if self.temporal_grounding_mode == "certificate_v2"
            else overlaps(old.valid_start, old.valid_end, new.valid_start, new.valid_end)
        )
        if not interval_overlap:
            if policy.stateful and _newer_valid_start(
                old, new, mode=self.temporal_relation_mode
            ):
                return RelationType.SUPERSEDES, "later non-overlapping state interval"
            return RelationType.COEXISTS, "non-overlapping valid-time intervals"
        if new.update_kind == UpdateKind.CORRECTION:
            return RelationType.CORRECTS, "explicit correction with overlapping validity"
        if new.update_kind == UpdateKind.TRANSITION and _newer_valid_start(
            old, new, mode=self.temporal_relation_mode
        ):
            return RelationType.SUPERSEDES, "explicit later state transition"
        if policy.functional:
            return (
                RelationType.CONTRADICTS,
                "functional slot has different values in overlapping scope",
            )
        return RelationType.COEXISTS, "non-functional predicate permits multiple values"

    def _materialize(
        self,
        old: Claim,
        new: Claim,
        relation: RelationType,
        outcome: ConflictOutcome,
    ) -> None:
        if relation == RelationType.CORRECTS:
            mutation = {"claim_id": old.id, "status": ClaimStatus.CORRECTED}
            if self.close_corrected_transaction:
                mutation["transaction_end"] = new.transaction_start
            outcome.mutations.append(mutation)
        elif relation == RelationType.RETRACTS:
            outcome.mutations.append(
                {
                    "claim_id": old.id,
                    "status": ClaimStatus.RETRACTED,
                    "transaction_end": new.transaction_start,
                }
            )
        elif relation == RelationType.SUPERSEDES:
            mutation = {"claim_id": old.id, "status": ClaimStatus.SUPERSEDED}
            if (
                self.infer_transition_end
                and new.valid_start is not None
                and (
                    self.temporal_grounding_mode != "certificate_v2"
                    or old.valid_start is None
                    or old.valid_start < new.valid_start
                )
                and (old.valid_end is None or old.valid_end > new.valid_start)
            ):
                mutation["valid_end"] = new.valid_start
            outcome.mutations.append(mutation)
        elif relation == RelationType.CONTRADICTS and old.status == ClaimStatus.ACTIVE:
            outcome.mutations.append({"claim_id": old.id, "status": ClaimStatus.DISPUTED})


def _newer_valid_start(old: Claim, new: Claim, *, mode: str) -> bool:
    if new.valid_start is None:
        if mode == "valid_interval_v2" and new.valid_end is not None:
            return False
        if mode not in {"transaction_fallback_v1", "valid_interval_v2"}:
            raise ValueError(f"unsupported temporal relation mode: {mode}")
        return new.assertion_time >= old.assertion_time
    old_start = evidence_interval(old)[0] if mode == "valid_interval_v2" else old.valid_start
    if old_start is None:
        return True
    return new.valid_start >= old_start


def _draft_is_newer(old: Claim, new: ClaimDraft, *, mode: str) -> bool:
    if new.valid_start is None:
        if mode == "valid_interval_v2" and new.valid_end is not None:
            return False
        if mode not in {"transaction_fallback_v1", "valid_interval_v2"}:
            raise ValueError(f"unsupported temporal relation mode: {mode}")
        return new.assertion_time >= old.assertion_time
    old_start = evidence_interval(old)[0] if mode == "valid_interval_v2" else old.valid_start
    if old_start is None:
        return True
    return new.valid_start >= old_start
