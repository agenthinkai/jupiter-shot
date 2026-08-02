"""
Jupiter Shot — Mesh Router
============================
Routes inference requests to the best available node based on:
  - Model availability
  - Node health and load
  - Region affinity
  - Latency (p50/p99)

Routing strategies:
  - least_loaded: Route to node with lowest load_factor
  - round_robin: Distribute requests evenly
  - latency_aware: Route to node with lowest p50 latency
  - region_affinity: Prefer nodes in the same region as the client

Usage (as a standalone proxy):
    python mesh/router.py --registry-url http://localhost:9000 --port 9001

Usage (as a library):
    from mesh.router import MeshRouter
    router = MeshRouter(registry_url="http://localhost:9000")
    node_url = await router.select_node("jupiter-dense-1b3")
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    LATENCY_AWARE = "latency_aware"
    REGION_AFFINITY = "region_affinity"


class MeshRouter:
    """
    Routes inference requests to the best available node.

    Fetches node list from the Registry, applies routing strategy,
    and returns the inference URL of the selected node.
    """

    def __init__(
        self,
        registry_url: str,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        preferred_region: Optional[str] = None,
        cache_ttl_seconds: int = 5,
    ) -> None:
        self.registry_url = registry_url
        self.strategy = strategy
        self.preferred_region = preferred_region
        self.cache_ttl_seconds = cache_ttl_seconds
        self._node_cache: list[dict] = []
        self._cache_updated_at: float = 0.0
        self._round_robin_index: dict[str, int] = {}

    async def _fetch_nodes(self, model_name: str) -> list[dict]:
        """Fetch available nodes for a model from the Registry."""
        now = time.time()
        if now - self._cache_updated_at < self.cache_ttl_seconds:
            # Return cached nodes filtered by model
            return [n for n in self._node_cache if n.get("info", {}).get("model_name") == model_name]

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.registry_url}/api/v1/nodes/model/{model_name}",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        nodes = data.get("nodes", [])
                        self._node_cache = nodes
                        self._cache_updated_at = now
                        return nodes
        except Exception as e:
            logger.warning(f"Could not fetch nodes from registry: {e}")

        return []

    def _select_least_loaded(self, nodes: list[dict]) -> Optional[dict]:
        """Select node with lowest load factor."""
        healthy = [n for n in nodes if n.get("status") == "healthy"]
        if not healthy:
            return None
        return min(
            healthy,
            key=lambda n: n.get("capacity", {}).get("load_factor", 1.0),
        )

    def _select_round_robin(self, nodes: list[dict], model_name: str) -> Optional[dict]:
        """Select next node in round-robin order."""
        healthy = [n for n in nodes if n.get("status") == "healthy"]
        if not healthy:
            return None
        idx = self._round_robin_index.get(model_name, 0)
        node = healthy[idx % len(healthy)]
        self._round_robin_index[model_name] = (idx + 1) % len(healthy)
        return node

    def _select_latency_aware(self, nodes: list[dict]) -> Optional[dict]:
        """Select node with lowest p50 latency."""
        healthy = [n for n in nodes if n.get("status") == "healthy"]
        if not healthy:
            return None
        return min(healthy, key=lambda n: n.get("p50_latency_ms", float("inf")))

    def _select_region_affinity(self, nodes: list[dict]) -> Optional[dict]:
        """Prefer nodes in the preferred region, fall back to least loaded."""
        if self.preferred_region:
            regional = [
                n for n in nodes
                if n.get("status") == "healthy"
                and n.get("info", {}).get("region") == self.preferred_region
            ]
            if regional:
                return self._select_least_loaded(regional)
        return self._select_least_loaded(nodes)

    async def select_node(self, model_name: str) -> Optional[str]:
        """
        Select the best node for a model and return its inference URL.

        Returns:
            Inference URL of the selected node, or None if no nodes available.
        """
        nodes = await self._fetch_nodes(model_name)
        if not nodes:
            logger.warning(f"No nodes available for model: {model_name}")
            return None

        if self.strategy == RoutingStrategy.LEAST_LOADED:
            node = self._select_least_loaded(nodes)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            node = self._select_round_robin(nodes, model_name)
        elif self.strategy == RoutingStrategy.LATENCY_AWARE:
            node = self._select_latency_aware(nodes)
        elif self.strategy == RoutingStrategy.REGION_AFFINITY:
            node = self._select_region_affinity(nodes)
        else:
            node = self._select_least_loaded(nodes)

        if node:
            url = node.get("info", {}).get("inference_url", "")
            logger.debug(f"Routing {model_name} → {node.get('info', {}).get('node_id')} @ {url}")
            return url

        return None

    async def proxy_request(
        self,
        model_name: str,
        path: str,
        method: str = "POST",
        body: Optional[dict] = None,
    ) -> tuple[int, dict]:
        """
        Proxy a request to the selected node.

        Returns:
            Tuple of (status_code, response_body).
        """
        node_url = await self.select_node(model_name)
        if not node_url:
            return 503, {"error": f"No nodes available for model: {model_name}"}

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    f"{node_url}{path}",
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    response_body = await resp.json()
                    return resp.status, response_body
        except Exception as e:
            logger.error(f"Proxy request failed: {e}")
            return 500, {"error": str(e)}


# ── FastAPI app wrapper for standalone deployment ─────────────────────────────

import os

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn as _uvicorn

    app = FastAPI(title="Jupiter Shot Mesh Router", version="0.1.0")
    _router_instance: Optional[MeshRouter] = None

    @app.on_event("startup")
    async def _startup():
        global _router_instance
        registry_url = os.environ.get("MESH_REGISTRY_URL", "http://localhost:9000")
        _router_instance = MeshRouter(registry_url=registry_url)

    @app.post("/v1/{path:path}")
    async def proxy(path: str, request: Request):
        model_name = path.split("/")[0]
        body = await request.json()
        status, resp = await _router_instance.proxy_request(
            model_name, f"/{path}", method=request.method, body=body
        )
        return JSONResponse(content=resp, status_code=status)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "mesh-router"}

except ImportError:
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jupiter Shot Mesh Router")
    parser.add_argument("--registry-url", default="http://localhost:9000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    os.environ["MESH_REGISTRY_URL"] = args.registry_url
    import uvicorn
    uvicorn.run("mesh.router:app", host=args.host, port=args.port, reload=False)
