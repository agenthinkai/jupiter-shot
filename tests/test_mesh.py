"""
Jupiter Shot — Mesh Unit Tests
================================
Tests for compliance logger, node agent, and registry.

Run:
    pytest tests/test_mesh.py -v
"""

import pytest
import asyncio
import json
import tempfile
import time
import uuid
from pathlib import Path


class TestComplianceLogger:
    @pytest.fixture
    def log_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            yield f.name

    def test_log_request(self, log_file):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(output_path=log_file, log_to_stdout=False)
        req_id = str(uuid.uuid4())
        cl.log_request(
            request_id=req_id,
            model_name="jupiter-dense-1b3",
            node_id="node-001",
            region="us-east-1",
            prompt="Hello world",
            max_tokens=50,
        )
        cl.close()

        with open(log_file) as f:
            record = json.loads(f.readline())

        assert record["event_type"] == "inference_request"
        assert record["request_id"] == req_id
        assert record["model_name"] == "jupiter-dense-1b3"
        assert record["node_id"] == "node-001"
        # Prompt should be hashed, not stored plaintext
        assert "prompt_hash" in record
        assert record["prompt_hash"] != "Hello world"
        assert len(record["prompt_hash"]) == 64  # SHA-256 hex

    def test_prompt_not_stored_plaintext(self, log_file):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(output_path=log_file, log_to_stdout=False)
        sensitive_prompt = "My SSN is 123-45-6789"
        cl.log_request(
            request_id="req-001",
            model_name="test-model",
            node_id="node-001",
            region="us-east-1",
            prompt=sensitive_prompt,
            max_tokens=50,
        )
        cl.close()

        content = Path(log_file).read_text()
        assert "123-45-6789" not in content, "PII found in log file!"
        assert "My SSN" not in content, "PII found in log file!"

    def test_log_response(self, log_file):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(output_path=log_file, log_to_stdout=False)
        req_id = str(uuid.uuid4())
        cl.log_response(
            request_id=req_id,
            model_name="jupiter-dense-1b3",
            node_id="node-001",
            completion_tokens=12,
            total_tokens=18,
            latency_ms=245.3,
            finish_reason="stop",
            output_text="Paris is the capital of France.",
            success=True,
        )
        cl.close()

        with open(log_file) as f:
            record = json.loads(f.readline())

        assert record["event_type"] == "inference_response"
        assert record["request_id"] == req_id
        assert record["completion_tokens"] == 12
        assert record["latency_ms"] == pytest.approx(245.3)
        assert record["success"] is True
        # Output should be hashed
        assert "output_hash" in record
        assert record["output_hash"] != "Paris is the capital of France."

    def test_log_node_event(self, log_file):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(output_path=log_file, log_to_stdout=False)
        cl.log_node_event(
            node_id="node-001",
            model_name="jupiter-dense-1b3",
            event="registered",
            region="us-east-1",
            details={"gpu_type": "a100_80gb"},
        )
        cl.close()

        with open(log_file) as f:
            record = json.loads(f.readline())

        assert record["event_type"] == "node_event"
        assert record["event"] == "registered"
        assert record["details"]["gpu_type"] == "a100_80gb"

    def test_hash_consistency(self):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(log_to_stdout=False)
        h1 = cl._hash("test string")
        h2 = cl._hash("test string")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_hash_different_inputs(self):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(log_to_stdout=False)
        h1 = cl._hash("input A")
        h2 = cl._hash("input B")
        assert h1 != h2

    def test_multiple_records_in_file(self, log_file):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(output_path=log_file, log_to_stdout=False)
        for i in range(5):
            cl.log_request(
                request_id=f"req-{i:03d}",
                model_name="test-model",
                node_id="node-001",
                region="us-east-1",
                prompt=f"prompt {i}",
                max_tokens=50,
            )
        cl.close()

        lines = Path(log_file).read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            record = json.loads(line)
            assert record["event_type"] == "inference_request"

    def test_timestamp_iso_present(self, log_file):
        from mesh.compliance_logger import ComplianceLogger
        cl = ComplianceLogger(output_path=log_file, log_to_stdout=False)
        cl.log_request(
            request_id="req-001",
            model_name="test-model",
            node_id="node-001",
            region="us-east-1",
            prompt="test",
            max_tokens=50,
        )
        cl.close()

        with open(log_file) as f:
            record = json.loads(f.readline())

        assert "timestamp_iso" in record
        assert "T" in record["timestamp_iso"]  # ISO 8601 format


class TestNodeCapacity:
    def test_available_slots(self):
        from mesh.node_agent import NodeCapacity
        cap = NodeCapacity(max_concurrent_requests=32, current_requests=10)
        assert cap.available_slots == 22

    def test_available_slots_no_overflow(self):
        from mesh.node_agent import NodeCapacity
        cap = NodeCapacity(max_concurrent_requests=32, current_requests=40)
        assert cap.available_slots == 0

    def test_load_factor(self):
        from mesh.node_agent import NodeCapacity
        cap = NodeCapacity(max_concurrent_requests=32, current_requests=16)
        assert cap.load_factor == pytest.approx(0.5)

    def test_load_factor_idle(self):
        from mesh.node_agent import NodeCapacity
        cap = NodeCapacity(max_concurrent_requests=32, current_requests=0)
        assert cap.load_factor == pytest.approx(0.0)


class TestNodeRegistry:
    @pytest.fixture
    def registry(self):
        from mesh.registry import NodeRegistry
        return NodeRegistry()

    @pytest.fixture
    def sample_node(self):
        from mesh.registry import NodeRegistration
        return NodeRegistration(
            node_id="test-node-001",
            model_name="jupiter-dense-1b3",
            model_type="dense",
            inference_url="http://localhost:8000",
            region="us-east-1",
            zone="us-east-1a",
            hostname="test-host",
            ip_address="10.0.0.1",
            gpu_type="a100_80gb",
            num_gpus=1,
        )

    @pytest.mark.asyncio
    async def test_register_node(self, registry, sample_node):
        await registry.register(sample_node)
        node = await registry.get_node("test-node-001")
        assert node is not None
        assert node.info.model_name == "jupiter-dense-1b3"

    @pytest.mark.asyncio
    async def test_deregister_node(self, registry, sample_node):
        await registry.register(sample_node)
        success = await registry.deregister("test-node-001")
        assert success is True
        node = await registry.get_node("test-node-001")
        assert node is None

    @pytest.mark.asyncio
    async def test_deregister_nonexistent(self, registry):
        success = await registry.deregister("nonexistent-node")
        assert success is False

    @pytest.mark.asyncio
    async def test_heartbeat_updates_status(self, registry, sample_node):
        from mesh.registry import HeartbeatPayload
        await registry.register(sample_node)
        payload = HeartbeatPayload(
            node_id="test-node-001",
            timestamp=time.time(),
            status="healthy",
            capacity={"load_factor": 0.3},
            p50_latency_ms=120.0,
        )
        success = await registry.heartbeat(payload)
        assert success is True

        node = await registry.get_node("test-node-001")
        assert node.p50_latency_ms == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_get_nodes_for_model(self, registry, sample_node):
        await registry.register(sample_node)
        nodes = await registry.get_nodes_for_model("jupiter-dense-1b3")
        assert len(nodes) == 1
        assert nodes[0].info.node_id == "test-node-001"

    @pytest.mark.asyncio
    async def test_get_nodes_for_wrong_model(self, registry, sample_node):
        await registry.register(sample_node)
        nodes = await registry.get_nodes_for_model("nonexistent-model")
        assert len(nodes) == 0

    @pytest.mark.asyncio
    async def test_stale_node_marked_offline(self, registry, sample_node):
        await registry.register(sample_node)
        # Manually set last heartbeat to old time
        registry._nodes["test-node-001"].last_heartbeat = time.time() - 120
        await registry.check_stale_nodes()
        node = await registry.get_node("test-node-001")
        assert node is not None
        assert node.status == "offline"

    @pytest.mark.asyncio
    async def test_summary(self, registry, sample_node):
        await registry.register(sample_node)
        summary = registry.summary()
        assert summary["total_nodes"] == 1
        assert summary["healthy_nodes"] == 1
        assert "jupiter-dense-1b3" in summary["registered_models"]
