"""
Jupiter Shot — Mesh Node Registry
===================================
Central registry for all active inference nodes in the Mesh.

Responsibilities:
  - Accept node registrations and deregistrations
  - Track node health via heartbeats
  - Mark nodes as offline after missed heartbeats
  - Provide node discovery API for the Router

Runs as a standalone FastAPI service.

Usage:
    python mesh/registry.py --port 9000

API:
    POST /api/v1/nodes/register          — Register a new node
    DELETE /api/v1/nodes/{node_id}       — Deregister a node
    POST /api/v1/nodes/{node_id}/heartbeat — Update node health
    GET  /api/v1/nodes                   — List all active nodes
    GET  /api/v1/nodes/{node_id}         — Get a specific node
    GET  /api/v1/nodes/model/{model_name} — Get nodes for a model
    GET  /health                         — Registry health check
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class NodeRegistration(BaseModel):
    node_id: str
    model_name: str
    model_type: str
    inference_url: str
    region: str
    zone: str
    hostname: str
    ip_address: str
    gpu_type: str
    num_gpus: int
    quantization: Optional[str] = None
    version: str = "0.1.0"
    registered_at: float = 0.0
    tags: dict = {}


class HeartbeatPayload(BaseModel):
    node_id: str
    timestamp: float
    status: str
    capacity: dict
    error_rate_1m: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class NodeRecord(BaseModel):
    info: NodeRegistration
    last_heartbeat: float
    status: str
    capacity: dict
    error_rate_1m: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    missed_heartbeats: int = 0


# ── Registry Store ────────────────────────────────────────────────────────────

class NodeRegistry:
    """In-memory node registry with heartbeat tracking."""

    HEARTBEAT_TIMEOUT_SECONDS = 60  # Mark offline after 60s without heartbeat
    STALE_NODE_TTL_SECONDS = 300    # Remove completely after 5 minutes offline

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRecord] = {}
        self._lock = asyncio.Lock()

    async def register(self, info: NodeRegistration) -> None:
        async with self._lock:
            self._nodes[info.node_id] = NodeRecord(
                info=info,
                last_heartbeat=time.time(),
                status="healthy",
                capacity={},
            )
            logger.info(f"Node registered: {info.node_id} ({info.model_name}) @ {info.inference_url}")

    async def deregister(self, node_id: str) -> bool:
        async with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"Node deregistered: {node_id}")
                return True
            return False

    async def heartbeat(self, payload: HeartbeatPayload) -> bool:
        async with self._lock:
            if payload.node_id not in self._nodes:
                logger.warning(f"Heartbeat from unknown node: {payload.node_id}")
                return False
            node = self._nodes[payload.node_id]
            node.last_heartbeat = payload.timestamp
            node.status = payload.status
            node.capacity = payload.capacity
            node.error_rate_1m = payload.error_rate_1m
            node.p50_latency_ms = payload.p50_latency_ms
            node.p99_latency_ms = payload.p99_latency_ms
            node.missed_heartbeats = 0
            return True

    async def get_all_nodes(self, include_offline: bool = False) -> List[NodeRecord]:
        async with self._lock:
            nodes = list(self._nodes.values())
        if not include_offline:
            nodes = [n for n in nodes if n.status not in ("offline", "draining")]
        return nodes

    async def get_node(self, node_id: str) -> Optional[NodeRecord]:
        async with self._lock:
            return self._nodes.get(node_id)

    async def get_nodes_for_model(self, model_name: str) -> List[NodeRecord]:
        all_nodes = await self.get_all_nodes()
        return [n for n in all_nodes if n.info.model_name == model_name]

    async def check_stale_nodes(self) -> None:
        """Background task: mark nodes offline if heartbeat missed."""
        now = time.time()
        async with self._lock:
            stale_ids = []
            for node_id, node in self._nodes.items():
                age = now - node.last_heartbeat
                if age > self.STALE_NODE_TTL_SECONDS:
                    stale_ids.append(node_id)
                elif age > self.HEARTBEAT_TIMEOUT_SECONDS and node.status != "offline":
                    node.status = "offline"
                    node.missed_heartbeats += 1
                    logger.warning(
                        f"Node {node_id} marked OFFLINE "
                        f"(last heartbeat {age:.0f}s ago)"
                    )
            for node_id in stale_ids:
                del self._nodes[node_id]
                logger.info(f"Removed stale node: {node_id}")

    def summary(self) -> dict:
        total = len(self._nodes)
        healthy = sum(1 for n in self._nodes.values() if n.status == "healthy")
        offline = sum(1 for n in self._nodes.values() if n.status == "offline")
        models = list({n.info.model_name for n in self._nodes.values()})
        return {
            "total_nodes": total,
            "healthy_nodes": healthy,
            "offline_nodes": offline,
            "registered_models": models,
        }


# ── FastAPI Application ───────────────────────────────────────────────────────

registry = NodeRegistry()


async def stale_node_checker():
    """Background task to check for stale nodes every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        await registry.check_stale_nodes()


app = FastAPI(
    title="Jupiter Shot Mesh Registry",
    description="Node registry for the Jupiter Shot Mesh",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    asyncio.create_task(stale_node_checker())
    logger.info("Mesh Registry started")


@app.get("/health")
async def health():
    return {"status": "healthy", **registry.summary()}


@app.post("/api/v1/nodes/register")
async def register_node(info: NodeRegistration):
    await registry.register(info)
    return {"status": "registered", "node_id": info.node_id}


@app.delete("/api/v1/nodes/{node_id}")
async def deregister_node(node_id: str):
    success = await registry.deregister(node_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return {"status": "deregistered", "node_id": node_id}


@app.post("/api/v1/nodes/{node_id}/heartbeat")
async def node_heartbeat(node_id: str, payload: HeartbeatPayload):
    if payload.node_id != node_id:
        raise HTTPException(status_code=400, detail="node_id mismatch")
    success = await registry.heartbeat(payload)
    if not success:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not registered")
    return {"status": "ok"}


@app.get("/api/v1/nodes")
async def list_nodes(include_offline: bool = False):
    nodes = await registry.get_all_nodes(include_offline=include_offline)
    return {"nodes": [n.dict() for n in nodes], "count": len(nodes)}


@app.get("/api/v1/nodes/{node_id}")
async def get_node(node_id: str):
    node = await registry.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return node.dict()


@app.get("/api/v1/nodes/model/{model_name}")
async def get_nodes_for_model(model_name: str):
    nodes = await registry.get_nodes_for_model(model_name)
    return {"model": model_name, "nodes": [n.dict() for n in nodes], "count": len(nodes)}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Jupiter Shot Mesh Registry")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
