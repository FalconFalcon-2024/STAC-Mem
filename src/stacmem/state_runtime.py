"""Typed scope isolation and positive day observations for state memory."""

from __future__ import annotations

from collections import defaultdict

from .admission import AdmissionAssessment, FactualityAdmissionGate
from .conflict import ConflictEngine
from .contracts import (
    ContractClaimExtractor,
    ContractPolicies,
    ContractQueryCompiler,
    scope_key,
)
from .models import AdmissionDecision, EvidencePack, RelationType
from .pipeline import StacMemory
from .place_schema import PlaceRegistry
from .resolver import StateResolver, compose_context
from .temporal_eligibility import evidence_overlaps
from .temporal_grounding import TemporalGrounder


class SourceAdmission(FactualityAdmissionGate):
    def __init__(self, *, policies=None, **kwargs):
        super().__init__(**kwargs)
        self.policies = policies or ContractPolicies()

    def assess(self, draft, *, existing=()):
        contract = draft.metadata.get("source_contract", {})
        if contract.get("status") != "validated":
            return AdmissionAssessment(
                AdmissionDecision.QUARANTINE,
                tuple(contract.get("reasons") or ["source_contract_missing"]),
                "source-admission-v1",
            )
        if draft.predicate not in self.policies.specs:
            return AdmissionAssessment(
                AdmissionDecision.QUARANTINE,
                ("predicate_not_in_state_schema",),
                "source-admission-v1",
            )
        return super().assess(draft, existing=existing)


class ObservationGrounder(TemporalGrounder):
    """Use the shared proposition-bound temporal observation contract."""


class ScopedConflictEngine(ConflictEngine):
    def _normalize_update_kind(self, draft, existing, policy):
        if draft.metadata.get("temporal_observation"):
            return
        if policy.spatially_scoped:
            existing = [
                c
                for c in existing
                if scope_key(c.place) == scope_key(draft.place) and scope_key(c.place)
            ]
        return super()._normalize_update_kind(draft, existing, policy)

    def _classify(self, old, new, policy):
        if policy.spatially_scoped:
            left, right = scope_key(old.place), scope_key(new.place)
            if left is None or right is None:
                return RelationType.COEXISTS, "unverified_scope_no_destructive_update"
            if left != right:
                return RelationType.COEXISTS, "distinct_literal_context_keys"
        if old.metadata.get("temporal_observation") or new.metadata.get("temporal_observation"):
            if not evidence_overlaps(old, new):
                return RelationType.COEXISTS, "observation_does_not_assert_state_transition"
            if old.object_norm != new.object_norm:
                return RelationType.CONTRADICTS, "different_values_at_observed_day"
        return super()._classify(old, new, policy)


class ScopedResolver(StateResolver):
    def __init__(self, store, *, policies, token_budget=4000):
        super().__init__(store, token_budget=token_budget)
        self.policies = policies

    def resolve(self, frame, candidates, *, variant, top_k=None):
        if not variant.conflict or not variant.spatial:
            return super().resolve(frame, candidates, variant=variant, top_k=top_k)
        groups = defaultdict(list)
        excluded = unknown = 0
        requested = scope_key(frame.place)
        for item in candidates:
            scoped = self.policies.resolve(
                item.claim.predicate, item.claim.functional
            ).spatially_scoped
            key = scope_key(item.claim.place) if scoped else None
            if scoped and key is None:
                unknown += 1
                continue
            if scoped and requested is not None and key != requested:
                excluded += 1
                continue
            groups[(item.claim.slot_key, key if scoped else None)].append(item)
        packs = [
            super(ScopedResolver, self).resolve(
                frame,
                items,
                variant=variant,
                top_k=top_k,
            )
            for items in groups.values()
        ]
        selected = sorted(
            [c for p in packs for c in p.claims], key=lambda c: c.score.total, reverse=True
        )
        warnings = list(dict.fromkeys(w for p in packs for w in p.warnings))
        if unknown:
            warnings.append("unverified place scope excluded; consult original memory")
        if (
            requested is None
            and len(
                {
                    scope_key(c.claim.place)
                    for c in selected
                    if self.policies.resolve(c.claim.predicate, c.claim.functional).spatially_scoped
                }
            )
            > 1
        ):
            warnings.append("multiple place scopes; specify a location before selecting one")
        pack = EvidencePack(
            query=frame,
            claims=selected[: top_k or 10],
            warnings=warnings,
            diagnostics={
                "resolver": "scoped-state-evidence-floor-v2",
                "input_candidates": len(candidates),
                "selected_claims": len(selected[: top_k or 10]),
                "scope_excluded": excluded,
                "scope_unverified": unknown,
                "suppressed_claims": sum(p.diagnostics.get("suppressed_claims", 0) for p in packs),
                "unresolved_slots": sum(p.diagnostics.get("unresolved_slots", 0) for p in packs),
                "query_routing": frame.metadata.get("retrieval_route", "unknown"),
            },
        )
        observations = [
            c.claim.id for c in pack.claims if c.claim.metadata.get("temporal_observation")
        ]
        footers = (
            ["Day observations are not inferred onset or termination events."]
            if observations
            else []
        )
        pack.context = compose_context(pack, self.token_budget, footer_lines=footers)
        return pack


def validate_contract_config(config) -> None:
    """Reject settings that the public state runtime cannot honor."""
    required = {
        "admission.enabled": (config.admission.enabled, True),
        "admission.mode": (config.admission.mode, "proposition_v2"),
        "temporal_grounding.enabled": (config.temporal_grounding.enabled, True),
        "temporal_grounding.mode": (
            config.temporal_grounding.mode,
            "certificate_v2",
        ),
        "conflict.transition_mode": (config.conflict.transition_mode, "proposition_v2"),
        "conflict.temporal_relation_mode": (
            config.conflict.temporal_relation_mode,
            "valid_interval_v2",
        ),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "The public state runtime has fixed correctness semantics: " + "; ".join(mismatches)
        )


def install_contracts(memory: StacMemory) -> StacMemory:
    """Install the default public state contracts after database compatibility checks."""
    config = memory.config
    validate_contract_config(config)
    policies = ContractPolicies(config.predicate_schema)
    places = PlaceRegistry(config.place_schema)
    memory.place_registry = places
    memory.conflict = ScopedConflictEngine(
        memory.store,
        policies=policies,
        admission_gate=SourceAdmission(
            enabled=config.admission.enabled,
            mode=config.admission.mode,
            policies=policies,
        ),
        temporal_grounder=ObservationGrounder(),
        temporal_grounding_mode=config.temporal_grounding.mode,
        temporal_relation_mode=config.conflict.temporal_relation_mode,
        transition_mode=config.conflict.transition_mode,
        close_corrected_transaction=config.conflict.close_corrected_transaction,
        infer_transition_end=config.conflict.infer_transition_end,
        infer_transition_from_evidence=config.conflict.infer_transition_from_evidence,
    )
    memory.resolver = ScopedResolver(
        memory.store, policies=policies, token_budget=config.retrieval.token_budget
    )
    if memory.claim_extractor is not None:
        memory.claim_extractor = ContractClaimExtractor(
            memory.claim_extractor.backend, policies=policies, place_registry=places
        )
        memory.query_compiler = ContractQueryCompiler(
            memory.query_compiler.backend, policies=policies, place_registry=places
        )
    return memory
