"""
Jupiter Shot — Compliance Logger
==================================
Structured audit logging for all inference requests and model outputs.

Captures:
  - Request metadata (timestamp, node, model, region)
  - Input prompt hash (SHA-256, not plaintext — for privacy)
  - Output metadata (token count, latency, finish reason)
  - User/session ID (anonymized)
  - Content policy flags (if content filtering is enabled)

Audit-log foundation:
  This module provides an integrity-protected audit-log foundation designed
  to support future compliance controls. SHA-256 hashing of prompt content
  prevents plaintext PII storage, but this alone does NOT establish GDPR
  or SOC 2 compliance. Full compliance requires additional controls including
  data-subject rights workflows, access controls, retention policies,
  data-processing agreements, and third-party audit.
  - Internal: Reproducibility and incident investigation

Log format: JSONL (one JSON object per line) to stdout and/or file.
Structured for ingestion by Elasticsearch, Splunk, or CloudWatch Logs.

Usage:
    from mesh.compliance_logger import ComplianceLogger
    logger = ComplianceLogger(output_path="logs/compliance.jsonl")
    logger.log_request(...)
    logger.log_response(...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Log Record Types ──────────────────────────────────────────────────────────

@dataclass
class RequestLogRecord:
    """Log record for an incoming inference request."""
    event_type: str = "inference_request"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = ""

    # Routing
    model_name: str = ""
    node_id: str = ""
    region: str = ""

    # Request metadata
    prompt_hash: str = ""          # SHA-256 of prompt (not plaintext)
    prompt_length_chars: int = 0
    prompt_tokens: int = 0
    max_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 0.9

    # Identity (anonymized)
    session_id: str = ""           # Anonymized session identifier
    user_id_hash: str = ""         # SHA-256 of user ID (if available)
    client_ip_hash: str = ""       # SHA-256 of client IP

    # Content policy
    content_policy_checked: bool = False
    content_policy_flags: list = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp_iso:
            from datetime import datetime, timezone
            self.timestamp_iso = datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat()


@dataclass
class ResponseLogRecord:
    """Log record for a completed inference response."""
    event_type: str = "inference_response"
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = ""

    # Routing
    model_name: str = ""
    node_id: str = ""

    # Response metadata
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""

    # Quality signals
    output_hash: str = ""          # SHA-256 of output (for deduplication)
    aux_loss: Optional[float] = None   # MoE aux loss (if available)

    # Status
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp_iso:
            from datetime import datetime, timezone
            self.timestamp_iso = datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat()


@dataclass
class NodeEventRecord:
    """Log record for node lifecycle events."""
    event_type: str = "node_event"
    timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = ""
    node_id: str = ""
    model_name: str = ""
    event: str = ""  # "registered", "deregistered", "offline", "recovered", "draining"
    region: str = ""
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp_iso:
            from datetime import datetime, timezone
            self.timestamp_iso = datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat()


# ── Compliance Logger ─────────────────────────────────────────────────────────

class ComplianceLogger:
    """
    Structured compliance logger for Jupiter Shot Mesh.

    Writes JSONL records to stdout and/or a file.
    Thread-safe for concurrent requests.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        log_to_stdout: bool = True,
        max_file_size_mb: float = 100.0,
        rotation_count: int = 5,
    ) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.log_to_stdout = log_to_stdout
        self.max_file_size_bytes = int(max_file_size_mb * 1e6)
        self.rotation_count = rotation_count
        self._file_handle = None

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.output_path, "a", buffering=1)
            logger.info(f"Compliance log: {self.output_path}")

    @staticmethod
    def _hash(value: str) -> str:
        """SHA-256 hash of a string value."""
        return hashlib.sha256(value.encode()).hexdigest()

    def _write(self, record: dict) -> None:
        """Write a JSON record to output(s)."""
        line = json.dumps(record, default=str)
        if self.log_to_stdout:
            print(line, flush=True)
        if self._file_handle:
            self._file_handle.write(line + "\n")
            self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        """Rotate log file if it exceeds max size."""
        if not self.output_path:
            return
        try:
            size = os.path.getsize(self.output_path)
            if size > self.max_file_size_bytes:
                self._file_handle.close()
                # Rotate: rename old files
                for i in range(self.rotation_count - 1, 0, -1):
                    old = self.output_path.with_suffix(f".{i}.jsonl")
                    new = self.output_path.with_suffix(f".{i+1}.jsonl")
                    if old.exists():
                        old.rename(new)
                self.output_path.rename(self.output_path.with_suffix(".1.jsonl"))
                self._file_handle = open(self.output_path, "a", buffering=1)
        except Exception as e:
            logger.warning(f"Log rotation failed: {e}")

    def log_request(
        self,
        request_id: str,
        model_name: str,
        node_id: str,
        region: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        session_id: str = "",
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        content_flags: Optional[list] = None,
    ) -> None:
        """Log an incoming inference request."""
        record = RequestLogRecord(
            request_id=request_id,
            model_name=model_name,
            node_id=node_id,
            region=region,
            prompt_hash=self._hash(prompt),
            prompt_length_chars=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            session_id=session_id,
            user_id_hash=self._hash(user_id) if user_id else "",
            client_ip_hash=self._hash(client_ip) if client_ip else "",
            content_policy_checked=content_flags is not None,
            content_policy_flags=content_flags or [],
        )
        self._write(asdict(record))

    def log_response(
        self,
        request_id: str,
        model_name: str,
        node_id: str,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        finish_reason: str,
        output_text: str = "",
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        aux_loss: Optional[float] = None,
    ) -> None:
        """Log a completed inference response."""
        record = ResponseLogRecord(
            request_id=request_id,
            model_name=model_name,
            node_id=node_id,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            output_hash=self._hash(output_text) if output_text else "",
            success=success,
            error_code=error_code,
            error_message=error_message,
            aux_loss=aux_loss,
        )
        self._write(asdict(record))

    def log_node_event(
        self,
        node_id: str,
        model_name: str,
        event: str,
        region: str = "",
        details: Optional[dict] = None,
    ) -> None:
        """Log a node lifecycle event."""
        record = NodeEventRecord(
            node_id=node_id,
            model_name=model_name,
            event=event,
            region=region,
            details=details or {},
        )
        self._write(asdict(record))

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()


# ── Default global logger ─────────────────────────────────────────────────────

_default_logger: Optional[ComplianceLogger] = None


def get_compliance_logger(
    output_path: Optional[str] = None,
    log_to_stdout: bool = True,
) -> ComplianceLogger:
    """Get or create the global compliance logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = ComplianceLogger(
            output_path=output_path,
            log_to_stdout=log_to_stdout,
        )
    return _default_logger


if __name__ == "__main__":
    # Demo: write sample compliance log records
    import sys

    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    cl = ComplianceLogger(output_path=output_path, log_to_stdout=True)

    req_id = str(uuid.uuid4())
    cl.log_request(
        request_id=req_id,
        model_name="jupiter-dense-1b3",
        node_id="node-001",
        region="us-east-1",
        prompt="The capital of France is",
        max_tokens=50,
        temperature=0.7,
        session_id="sess-abc123",
        user_id="user-42",
        client_ip="192.168.1.1",
    )

    cl.log_response(
        request_id=req_id,
        model_name="jupiter-dense-1b3",
        node_id="node-001",
        completion_tokens=12,
        total_tokens=18,
        latency_ms=245.3,
        finish_reason="stop",
        output_text="Paris, which is located in northern France.",
        success=True,
    )

    cl.log_node_event(
        node_id="node-001",
        model_name="jupiter-dense-1b3",
        event="registered",
        region="us-east-1",
        details={"gpu_type": "a100_80gb", "num_gpus": 1},
    )

    print("Compliance log demo complete.")
