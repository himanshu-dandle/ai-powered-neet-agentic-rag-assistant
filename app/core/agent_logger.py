from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentStep:
    """
    A single traceable step performed by an agent.

    This is what we will show in Streamlit to prove the system is agentic:
    - what the agent decided
    - why it decided it (summary)
    - what it produced
    - confidence
    - timing
    """

    step_id: str
    request_id: str
    agent_name: str
    action: str

    inputs_summary: str
    outputs_summary: str

    confidence: float
    elapsed_ms: int

    meta: Dict[str, Any]


class AgentLogger:
    """
    Collects agent steps in-memory for the UI and optionally writes them to disk.

    Production-friendly:
    - request_id allows tracing one user request across multiple agents
    - JSONL logs are easy to ship to ELK / Datadog later
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self.request_id = str(uuid.uuid4())
        self._steps: List[AgentStep] = []
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def steps(self) -> List[AgentStep]:
        return self._steps

    def log_step(
        self,
        *,
        agent_name: str,
        action: str,
        inputs_summary: str,
        outputs_summary: str,
        confidence: float,
        elapsed_ms: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        step = AgentStep(
            step_id=str(uuid.uuid4()),
            request_id=self.request_id,
            agent_name=agent_name,
            action=action,
            inputs_summary=inputs_summary[:1500],
            outputs_summary=outputs_summary[:2000],
            confidence=float(confidence),
            elapsed_ms=int(elapsed_ms),
            meta=meta or {},
        )
        self._steps.append(step)

    def flush_to_disk(self) -> str:
        """
        Write steps to a JSONL file (one JSON object per line).
        Returns the written file path for debugging.
        """
        filename = f"agent_trace_{self.request_id}.jsonl"
        path = self._log_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for step in self._steps:
                f.write(json.dumps(asdict(step), ensure_ascii=False) + "\n")
        return str(path)


class timed:
    """
    Tiny helper context manager to measure elapsed time in ms.

    Usage:
        with timed() as t:
            do_work()
        elapsed = t.elapsed_ms
    """

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        self.elapsed_ms = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
