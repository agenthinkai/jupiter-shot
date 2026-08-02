"""
Jupiter Shot — Mesh Node Agent
================================
Each inference node runs a NodeAgent that:
  1. Registers itself with the Mesh Registry on startup
  2. Reports health and capacity metrics every N seconds
  3. Receives routing decisions from the Mesh Router
  4. Handles graceful shutdown (deregistration + drain)

The NodeAgent is the bridge between the inference server and the Mesh control plane.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Inference Node                                          │
  │  ┌──────────────┐    ┌──────────────┐                   │
  │  │  NodeAgent   │◄──►│ InferServer  │                   │
  │  │  (this file) │    │ (server.py)  │                   │
  │  └──────┬───────┘    └──────────────┘                   │
  │         │ HTTP/gRPC                                      │
  └─────────┼───────────────────────────────────────────────┘
            │
  ┌─────────▼───────────────────────────────────────────────┐
  │  Mesh Control Plane                                      │
  │  ┌──────────────┐    ┌──────────────┐                   │
  │  │   Registry   │    │    Router    │                   │
  │  │ (registry.py)│    │  (router.py) │                   │
  │  └──────────────┘    └──────────────┘                   │
  └─────────────────────────────────────────────────────────┘

Usage:
    python mesh/node_agent.py \
        --node-id node-001 \
        --model-name jupiter-dense-1b3 \
        --inference-url http://localhost:8000 \
        --registry-url http://mesh-registry:9000 \
        --region us-east-1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ── Node Metadata ─────────────────────────────────────────────────────────────

@dataclass
class NodeCapacity:
    """Current capacity and load of a node."""
    max_concurrent_requests: int = 32
    current_requests: int = 0
    queue_depth: int = 0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 80.0
    gpu_utilization_pct: float = 0.0
    tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent_requests - self.current_requests)

    @property
    def load_factor(self) -> float:
        """0.0 = idle, 1.0 = fully loaded."""
        return self.current_requests / max(1, self.max_concurrent_requests)


@dataclass
class NodeInfo:
    """Static information about a node."""
    node_id: str
    model_name: str
    model_type: str  # "dense" or "moe"
    inference_url: str
    region: str
    zone: str
    hostname: str
    ip_address: str
    gpu_type: str
    num_gpus: int
    quantization: Optional[str]  # "int8", "int4", or None
    version: str = "0.1.0"
    registered_at: float = field(default_factory=time.time)
    tags: dict = field(default_factory=dict)


@dataclass
class NodeHeartbeat:
    """Periodic heartbeat from node to registry."""
    node_id: str
    timestamp: float
    status: str  # "healthy", "degraded", "draining", "offline"
    capacity: NodeCapacity
    last_request_at: Optional[float] = None
    error_rate_1m: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


# ── Node Agent ────────────────────────────────────────────────────────────────

class NodeAgent:
    """
    Mesh node agent: registers and reports health to the Mesh Registry.

    Runs as a background coroutine alongside the inference server.
    """

    def __init__(
        self,
        node_id: str,
        model_name: str,
        model_type: str,
        inference_url: str,
        registry_url: str,
        region: str = "us-east-1",
        zone: str = "us-east-1a",
        gpu_type: str = "a100_80gb",
        num_gpus: int = 1,
        quantization: Optional[str] = None,
        heartbeat_interval: int = 15,
        max_concurrent_requests: int = 32,
    ) -> None:
        self.node_info = NodeInfo(
            node_id=node_id,
            model_name=model_name,
            model_type=model_type,
            inference_url=inference_url,
            region=region,
            zone=zone,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            quantization=quantization,
        )
        self.registry_url = registry_url
        self.heartbeat_interval = heartbeat_interval
        self.capacity = NodeCapacity(
            max_concurrent_requests=max_concurrent_requests,
            gpu_memory_total_gb=80.0 if "80gb" in gpu_type else 40.0,
        )
        self._status = "healthy"
        self._running = False
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._latency_samples: list[float] = []

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _get_gpu_metrics(self) -> dict:
        """Get current GPU utilization and memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return {"memory_used_gb": mem_used, "memory_total_gb": mem_total, "utilization_pct": 0.0}
        except Exception:
            pass
        return {"memory_used_gb": 0.0, "memory_total_gb": 80.0, "utilization_pct": 0.0}

    def _build_heartbeat(self) -> NodeHeartbeat:
        """Build a heartbeat payload."""
        gpu = self._get_gpu_metrics()
        self.capacity.gpu_memory_used_gb = gpu["memory_used_gb"]
        self.capacity.gpu_memory_total_gb = gpu["memory_total_gb"]
        self.capacity.gpu_utilization_pct = gpu["utilization_pct"]
        self.capacity.uptime_seconds = time.time() - self._start_time

        # Compute error rate and latency percentiles
        error_rate = self._error_count / max(1, self._request_count)
        latencies = sorted(self._latency_samples[-100:])  # Last 100 samples
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        return NodeHeartbeat(
            node_id=self.node_info.node_id,
            timestamp=time.time(),
            status=self._status,
            capacity=self.capacity,
            error_rate_1m=error_rate,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
        )

    async def _register(self) -> bool:
        """Register this node with the Mesh Registry."""
        try:
            import aiohttp
            payload = asdict(self.node_info)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/api/v1/nodes/register",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Registered with Mesh Registry: {self.registry_url}")
                        return True
                    else:
                        body = await resp.text()
                        logger.error(f"Registration failed: HTTP {resp.status} — {body}")
                        return False
        except Exception as e:
            logger.warning(f"Could not reach Mesh Registry at {self.registry_url}: {e}")
            logger.info("Running in standalone mode (no registry)")
            return False

    async def _send_heartbeat(self) -> bool:
        """Send a heartbeat to the Mesh Registry."""
        heartbeat = self._build_heartbeat()
        try:
            import aiohttp
            payload = {
                "node_id": heartbeat.node_id,
                "timestamp": heartbeat.timestamp,
                "status": heartbeat.status,
                "capacity": asdict(heartbeat.capacity),
                "error_rate_1m": heartbeat.error_rate_1m,
                "p50_latency_ms": heartbeat.p50_latency_ms,
                "p99_latency_ms": heartbeat.p99_latency_ms,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/api/v1/nodes/{self.node_info.node_id}/heartbeat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
            return False

    async def _deregister(self) -> None:
        """Deregister from the Mesh Registry on shutdown."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.delete(
                    f"{self.registry_url}/api/v1/nodes/{self.node_info.node_id}",
                    timeout=aiohttp.ClientTimeout(total=5),
                )
            logger.info(f"Deregistered from Mesh Registry")
        except Exception as e:
            logger.debug(f"Deregistration failed: {e}")

    async def run(self) -> None:
        """Main agent loop: register, then send heartbeats."""
        self._running = True
        self._start_time = time.time()

        logger.info(f"Node agent starting: {self.node_info.node_id}")
        logger.info(f"Model: {self.node_info.model_name} ({self.node_info.model_type})")
        logger.info(f"Inference URL: {self.node_info.inference_url}")
        logger.info(f"Registry URL: {self.registry_url}")

        # Register
        await self._register()

        # Heartbeat loop
        consecutive_failures = 0
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            success = await self._send_heartbeat()
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning(
                        f"Registry unreachable for {consecutive_failures} heartbeats. "
                        "Continuing in standalone mode."
                    )

        await self._deregister()
        logger.info("Node agent stopped.")

    def stop(self) -> None:
        """Signal the agent to stop."""
        self._running = False
        self._status = "offline"

    def set_draining(self) -> None:
        """Signal that this node is draining (no new requests)."""
        self._status = "draining"
        logger.info(f"Node {self.node_info.node_id} set to DRAINING")

    def record_request(self, latency_ms: float, success: bool) -> None:
        """Record a completed request for metrics."""
        self._request_count += 1
        if not success:
            self._error_count += 1
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 1000:
            self._latency_samples = self._latency_samples[-500:]

    def get_status(self) -> dict:
        """Return current agent status."""
        return {
            "node_id": self.node_info.node_id,
            "status": self._status,
            "model": self.node_info.model_name,
            "region": self.node_info.region,
            "uptime_seconds": time.time() - self._start_time,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "capacity": asdict(self.capacity),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jupiter Shot Mesh Node Agent")
    parser.add_argument("--node-id", default=f"node-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--model-name", default="jupiter-dense-1b3")
    parser.add_argument("--model-type", default="dense", choices=["dense", "moe"])
    parser.add_argument("--inference-url", default="http://localhost:8000")
    parser.add_argument("--registry-url", default="http://localhost:9000")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--zone", default="us-east-1a")
    parser.add_argument("--gpu-type", default="a100_80gb")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--heartbeat-interval", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    agent = NodeAgent(
        node_id=args.node_id,
        model_name=args.model_name,
        model_type=args.model_type,
        inference_url=args.inference_url,
        registry_url=args.registry_url,
        region=args.region,
        zone=args.zone,
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        quantization=args.quantization,
        heartbeat_interval=args.heartbeat_interval,
    )

    def handle_sigterm(signum, frame):
        logger.info("SIGTERM received — draining node")
        agent.set_draining()
        asyncio.get_event_loop().call_later(30, agent.stop)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, lambda s, f: agent.stop())

    asyncio.run(agent.run())
