"""
Jupiter Shot — Distributed Expert Registry

Tracks the location and health of expert shards across the serving cluster.
At 20T scale with 512 experts, each expert is served by a dedicated set of GPUs.
The registry maps expert IDs to serving endpoints and tracks load/health.

This is a reference implementation. Production deployment would use a
distributed key-value store (etcd, Redis Cluster) for the registry backend.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExpertStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # Responding but with elevated latency
    UNAVAILABLE = "unavailable" # Not responding
    LOADING = "loading"         # Weights being loaded into GPU memory
    DRAINING = "draining"       # Accepting no new requests, finishing current


@dataclass
class ExpertEndpoint:
    """A single serving endpoint for an expert shard."""
    endpoint_id: str
    host: str
    port: int
    gpu_ids: list[int]          # GPU indices on this host
    expert_ids: list[int]       # Expert IDs served by this endpoint
    model_stage: int            # Which Jupiter Shot stage (1-6)
    status: ExpertStatus = ExpertStatus.UNAVAILABLE
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    current_requests: int = 0
    total_requests_served: int = 0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.status == ExpertStatus.HEALTHY

    @property
    def load_factor(self) -> float:
        """Normalized load factor in [0, 1]. Higher = more loaded."""
        if self.vram_total_gb == 0:
            return 0.0
        vram_load = self.vram_used_gb / self.vram_total_gb
        request_load = min(1.0, self.current_requests / 100.0)  # Normalize to 100 concurrent
        return 0.7 * vram_load + 0.3 * request_load

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["address"] = self.address
        d["is_healthy"] = self.is_healthy
        d["load_factor"] = self.load_factor
        return d


@dataclass
class ExpertRegistration:
    """Registration record for an expert endpoint."""
    registration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint: ExpertEndpoint = field(default_factory=lambda: ExpertEndpoint(
        endpoint_id="", host="", port=0, gpu_ids=[], expert_ids=[], model_stage=1
    ))
    registered_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class ExpertRegistry:
    """
    In-memory expert registry. Tracks all expert endpoints across the serving cluster.

    For production use, replace the in-memory store with a distributed
    key-value store (etcd, Redis Cluster, or Consul).
    """

    # Heartbeat timeout: endpoints not heard from in this many seconds are marked unavailable
    HEARTBEAT_TIMEOUT_SECONDS = 30.0

    def __init__(self) -> None:
        self._endpoints: dict[str, ExpertEndpoint] = {}  # endpoint_id → endpoint
        self._expert_to_endpoints: dict[int, list[str]] = {}  # expert_id → [endpoint_ids]
        self._stage_to_endpoints: dict[int, list[str]] = {}   # model_stage → [endpoint_ids]

    def register(self, endpoint: ExpertEndpoint) -> str:
        """Register an expert endpoint. Returns the endpoint_id."""
        self._endpoints[endpoint.endpoint_id] = endpoint

        # Update expert → endpoint mapping
        for expert_id in endpoint.expert_ids:
            if expert_id not in self._expert_to_endpoints:
                self._expert_to_endpoints[expert_id] = []
            if endpoint.endpoint_id not in self._expert_to_endpoints[expert_id]:
                self._expert_to_endpoints[expert_id].append(endpoint.endpoint_id)

        # Update stage → endpoint mapping
        stage = endpoint.model_stage
        if stage not in self._stage_to_endpoints:
            self._stage_to_endpoints[stage] = []
        if endpoint.endpoint_id not in self._stage_to_endpoints[stage]:
            self._stage_to_endpoints[stage].append(endpoint.endpoint_id)

        logger.info(
            "Registered endpoint %s at %s serving experts %s (stage %d)",
            endpoint.endpoint_id, endpoint.address, endpoint.expert_ids, endpoint.model_stage
        )
        return endpoint.endpoint_id

    def deregister(self, endpoint_id: str) -> bool:
        """Deregister an endpoint. Returns True if found and removed."""
        if endpoint_id not in self._endpoints:
            return False

        endpoint = self._endpoints.pop(endpoint_id)

        for expert_id in endpoint.expert_ids:
            if expert_id in self._expert_to_endpoints:
                self._expert_to_endpoints[expert_id] = [
                    eid for eid in self._expert_to_endpoints[expert_id]
                    if eid != endpoint_id
                ]

        stage = endpoint.model_stage
        if stage in self._stage_to_endpoints:
            self._stage_to_endpoints[stage] = [
                eid for eid in self._stage_to_endpoints[stage]
                if eid != endpoint_id
            ]

        logger.info("Deregistered endpoint %s", endpoint_id)
        return True

    def heartbeat(self, endpoint_id: str, health_data: dict) -> bool:
        """
        Update endpoint health from a heartbeat. Returns True if endpoint found.

        health_data keys: status, vram_used_gb, vram_total_gb, current_requests,
                          p50_latency_ms, p99_latency_ms, error_count
        """
        if endpoint_id not in self._endpoints:
            return False

        ep = self._endpoints[endpoint_id]
        ep.last_heartbeat = time.time()
        ep.status = ExpertStatus(health_data.get("status", ExpertStatus.HEALTHY.value))
        ep.vram_used_gb = health_data.get("vram_used_gb", ep.vram_used_gb)
        ep.vram_total_gb = health_data.get("vram_total_gb", ep.vram_total_gb)
        ep.current_requests = health_data.get("current_requests", ep.current_requests)
        ep.p50_latency_ms = health_data.get("p50_latency_ms", ep.p50_latency_ms)
        ep.p99_latency_ms = health_data.get("p99_latency_ms", ep.p99_latency_ms)
        ep.error_count = health_data.get("error_count", ep.error_count)
        return True

    def get_endpoint_for_expert(
        self,
        expert_id: int,
        model_stage: int,
        strategy: str = "least_loaded",
    ) -> Optional[ExpertEndpoint]:
        """
        Get the best endpoint for serving a given expert.

        strategy: "least_loaded" | "round_robin" | "random"
        """
        self._mark_stale_endpoints()

        endpoint_ids = self._expert_to_endpoints.get(expert_id, [])
        candidates = [
            self._endpoints[eid]
            for eid in endpoint_ids
            if eid in self._endpoints
            and self._endpoints[eid].model_stage == model_stage
            and self._endpoints[eid].is_healthy
        ]

        if not candidates:
            logger.warning(
                "No healthy endpoint found for expert %d (stage %d)",
                expert_id, model_stage
            )
            return None

        if strategy == "least_loaded":
            return min(candidates, key=lambda ep: ep.load_factor)
        elif strategy == "round_robin":
            # Simple round-robin using total_requests_served as counter
            return min(candidates, key=lambda ep: ep.total_requests_served)
        else:
            import random
            return random.choice(candidates)

    def get_all_endpoints_for_stage(self, model_stage: int) -> list[ExpertEndpoint]:
        """Get all registered endpoints for a given model stage."""
        endpoint_ids = self._stage_to_endpoints.get(model_stage, [])
        return [self._endpoints[eid] for eid in endpoint_ids if eid in self._endpoints]

    def get_healthy_endpoints_for_stage(self, model_stage: int) -> list[ExpertEndpoint]:
        """Get all healthy endpoints for a given model stage."""
        self._mark_stale_endpoints()
        return [ep for ep in self.get_all_endpoints_for_stage(model_stage) if ep.is_healthy]

    def get_coverage(self, model_stage: int, num_experts: int) -> dict:
        """
        Check how many experts have at least one healthy endpoint.
        Returns coverage statistics.
        """
        self._mark_stale_endpoints()
        covered_experts = set()
        for expert_id in range(num_experts):
            endpoint_ids = self._expert_to_endpoints.get(expert_id, [])
            for eid in endpoint_ids:
                if eid in self._endpoints and self._endpoints[eid].is_healthy:
                    if self._endpoints[eid].model_stage == model_stage:
                        covered_experts.add(expert_id)
                        break

        return {
            "model_stage": model_stage,
            "num_experts": num_experts,
            "covered_experts": len(covered_experts),
            "coverage_pct": 100.0 * len(covered_experts) / num_experts if num_experts > 0 else 0.0,
            "missing_experts": [i for i in range(num_experts) if i not in covered_experts],
        }

    def get_registry_summary(self) -> dict:
        """Return a summary of the registry state."""
        self._mark_stale_endpoints()
        total = len(self._endpoints)
        healthy = sum(1 for ep in self._endpoints.values() if ep.is_healthy)
        stages = sorted(self._stage_to_endpoints.keys())

        return {
            "total_endpoints": total,
            "healthy_endpoints": healthy,
            "unavailable_endpoints": total - healthy,
            "stages_registered": stages,
            "endpoints_by_stage": {
                stage: len(ids) for stage, ids in self._stage_to_endpoints.items()
            },
        }

    def _mark_stale_endpoints(self) -> None:
        """Mark endpoints that haven't sent a heartbeat as unavailable."""
        now = time.time()
        for ep in self._endpoints.values():
            if (now - ep.last_heartbeat) > self.HEARTBEAT_TIMEOUT_SECONDS:
                if ep.status not in (ExpertStatus.UNAVAILABLE, ExpertStatus.LOADING):
                    logger.warning(
                        "Endpoint %s at %s timed out (last heartbeat %.0fs ago)",
                        ep.endpoint_id, ep.address, now - ep.last_heartbeat
                    )
                    ep.status = ExpertStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ExpertRegistry] = None


def get_registry() -> ExpertRegistry:
    """Get or create the module-level registry singleton."""
    global _registry
    if _registry is None:
        _registry = ExpertRegistry()
    return _registry
