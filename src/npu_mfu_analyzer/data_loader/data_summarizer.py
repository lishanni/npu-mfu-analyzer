"""
数据摘要化模块

将 GB 级 Profiling 数据转换为 KB 级摘要，适合 LLM 分析。
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class OverlapMetrics:
    """Overlap 指标"""
    total_compute_time: float = 0.0  # us
    total_comm_time: float = 0.0     # us
    overlapped_time: float = 0.0     # us
    comm_not_overlapped: float = 0.0 # us
    free_time: float = 0.0           # us
    overlap_ratio: float = 0.0       # %
    e2e_time: float = 0.0            # us


@dataclass
class StepMetrics:
    """单个 Step 的指标"""
    step: int = 0
    computing: float = 0.0
    communication: float = 0.0
    comm_not_overlapped: float = 0.0
    overlapped: float = 0.0
    free: float = 0.0
    stage: float = 0.0
    bubble: float = 0.0


@dataclass
class ProfilingSummary:
    """Profiling 数据摘要"""
    data_path: str = ""
    data_type: str = ""
    framework: str = ""
    rank_count: int = 0
    step_count: int = 0

    avg_step_time: float = 0.0
    avg_compute_time: float = 0.0
    avg_comm_time: float = 0.0
    avg_free_time: float = 0.0
    comm_ratio: float = 0.0

    overlap_metrics: OverlapMetrics = field(default_factory=OverlapMetrics)

    avg_bubble_time: float = 0.0
    bubble_ratio: float = 0.0

    time_breakdown: Dict[str, float] = field(default_factory=dict)
    top_operators: List[Dict[str, Any]] = field(default_factory=list)

    device_memory_gb: float = 64.0
    peak_memory_bytes: float = 0.0
    peak_memory_mb: float = 0.0
    peak_memory_gb: float = 0.0
    model_memory_mb: float = 0.0
    optimizer_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    gradient_memory_mb: float = 0.0
    temp_memory_mb: float = 0.0
    training_hints: Dict[str, Any] = field(default_factory=dict)

    sample_steps: List[StepMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["overlap_metrics"] = asdict(self.overlap_metrics)
        result["sample_steps"] = [asdict(s) for s in self.sample_steps]
        return result

    def to_prompt_text(self) -> str:
        lines = [
            "## Profiling 数据摘要",
            "",
            f"- 数据路径: {self.data_path}",
            f"- 数据类型: {self.data_type}",
            f"- 框架: {self.framework}",
            f"- Rank 数量: {self.rank_count}",
            f"- Step 数量: {self.step_count}",
            "",
            "### 时间指标（平均值）",
            f"- 平均 Step 时间: {self.avg_step_time / 1000:.2f} ms",
            f"- 计算时间: {self.avg_compute_time / 1000:.2f} ms ({self._ratio(self.avg_compute_time, self.avg_step_time):.1f}%)",
            f"- 通信时间: {self.avg_comm_time / 1000:.2f} ms ({self._ratio(self.avg_comm_time, self.avg_step_time):.1f}%)",
            f"- 空闲时间: {self.avg_free_time / 1000:.2f} ms ({self._ratio(self.avg_free_time, self.avg_step_time):.1f}%)",
            "",
            "### Overlap 分析",
            f"- 通信掩盖率: {self.overlap_metrics.overlap_ratio:.1f}%",
            f"- 未掩盖通信: {self.overlap_metrics.comm_not_overlapped / 1000:.2f} ms",
            f"- 已掩盖通信: {self.overlap_metrics.overlapped_time / 1000:.2f} ms",
        ]

        if self.avg_bubble_time > 0:
            lines.extend([
                "",
                "### Pipeline Bubble 分析",
                f"- 平均 Bubble 时间: {self.avg_bubble_time / 1000:.2f} ms",
                f"- Bubble 占比: {self.bubble_ratio:.1f}%",
            ])

        if self.peak_memory_gb > 0 or self.training_hints:
            lines.extend([
                "",
                "### 显存诊断线索",
                f"- 设备显存: {self.device_memory_gb:.0f} GB",
            ])
            if self.peak_memory_gb > 0:
                lines.append(f"- 峰值显存: {self.peak_memory_gb:.2f} GB")
            hints = self.training_hints or {}
            if hints.get("long_context_likely"):
                lines.append("- 训练形态提示: 更像长序列 dense")
            if hints.get("moe_likely"):
                lines.append("- 训练形态提示: 更像 MoE")
            if hints.get("activation_pressure_score", 0) > 0:
                lines.append(f"- Activation 压力分数: {hints.get('activation_pressure_score', 0):.2f}")
            if hints.get("state_pressure_score", 0) > 0:
                lines.append(f"- State 压力分数: {hints.get('state_pressure_score', 0):.2f}")

        if self.top_operators:
            lines.extend(["", "### Top 10 耗时算子"])
            for i, op in enumerate(self.top_operators[:10], 1):
                lines.append(f"{i}. {op.get('name', 'unknown')}: {op.get('dur', 0) / 1000:.2f} ms")

        return "\n".join(lines)

    def _ratio(self, part: float, total: float) -> float:
        return (part / total * 100) if total > 0 else 0


class DataSummarizer:
    _ATTENTION_PATTERNS = [
        r"attention", r"flashattention", r"inferattentionscore", r"scaled?masksoftmax",
        r"softmax", r"qkv", r"crossentropy", r"vocab", r"embedding",
    ]
    _ACTIVATION_AUX_PATTERNS = [r"layernorm", r"rmsnorm", r"dropout", r"gelu", r"silu", r"swiglu"]
    _OPTIMIZER_PATTERNS = [r"adam", r"lamb", r"optimizer", r"apply", r"update", r"moment"]
    _MOE_PATTERNS = [r"moe", r"expert", r"router", r"dispatch", r"combine", r"alltoall"]
    _PIPELINE_PATTERNS = [r"send", r"recv", r"pipeline", r"p2p"]

    def __init__(self, loader):
        self.loader = loader

    def summarize(self, max_sample_steps: int = 10) -> ProfilingSummary:
        info = self.loader.detect()

        summary = ProfilingSummary(
            data_path=info.path,
            data_type=info.data_type,
            framework=info.framework,
            rank_count=info.rank_count,
        )

        step_trace = self.loader.get_step_trace()
        if not step_trace.empty:
            summary.step_count = len(step_trace)
            summary.avg_compute_time = step_trace.get("computing", [0]).mean()
            summary.avg_comm_time = step_trace.get("communication", [0]).mean()
            summary.avg_free_time = step_trace.get("free", [0]).mean()

            if "computing" in step_trace.columns:
                free_col = step_trace["free"] if "free" in step_trace.columns else 0
                if "communication_not_overlapped" in step_trace.columns:
                    comm_col = step_trace["communication_not_overlapped"]
                elif "communication" in step_trace.columns:
                    comm_col = step_trace["communication"]
                else:
                    comm_col = 0
                step_trace = step_trace.assign(step_time=step_trace["computing"] + comm_col + free_col)
                summary.avg_step_time = step_trace["step_time"].mean()
                if summary.avg_step_time > 0:
                    summary.comm_ratio = summary.avg_comm_time / summary.avg_step_time

            if "communication_not_overlapped" in step_trace.columns:
                summary.overlap_metrics.comm_not_overlapped = step_trace["communication_not_overlapped"].mean()
            if "overlapped" in step_trace.columns:
                summary.overlap_metrics.overlapped_time = step_trace["overlapped"].mean()

            total_comm = summary.overlap_metrics.comm_not_overlapped + summary.overlap_metrics.overlapped_time
            if total_comm > 0:
                summary.overlap_metrics.overlap_ratio = (summary.overlap_metrics.overlapped_time / total_comm) * 100

            if "bubble" in step_trace.columns:
                summary.avg_bubble_time = step_trace["bubble"].mean()
                if "stage" in step_trace.columns:
                    avg_stage = step_trace["stage"].mean()
                    if avg_stage > 0:
                        summary.bubble_ratio = (summary.avg_bubble_time / avg_stage) * 100

            sample_indices = step_trace.index[:max_sample_steps]
            for idx in sample_indices:
                row = step_trace.iloc[idx]
                step_val = row.get("step", idx)
                step_int = int(step_val) if step_val == step_val else idx
                summary.sample_steps.append(StepMetrics(
                    step=step_int,
                    computing=float(row.get("computing", 0) or 0),
                    communication=float(row.get("communication", 0) or 0),
                    comm_not_overlapped=float(row.get("communication_not_overlapped", 0) or 0),
                    overlapped=float(row.get("overlapped", 0) or 0),
                    free=float(row.get("free", 0) or 0),
                    stage=float(row.get("stage", 0) or 0),
                    bubble=float(row.get("bubble", 0) or 0),
                ))

        timeline_summary = self.loader.get_timeline_summary()
        if timeline_summary:
            summary.time_breakdown = timeline_summary.get("by_category", {})

        summary.top_operators = self.loader.get_top_kernels(rank=None, top_n=10)

        hardware_info = self.loader.get_hardware_info()
        summary.device_memory_gb = self._extract_device_memory_gb(hardware_info)
        summary.training_hints = self._infer_training_hints(
            top_operators=summary.top_operators,
            time_breakdown=summary.time_breakdown,
            bubble_ratio=summary.bubble_ratio,
        )

        return summary

    def _extract_device_memory_gb(self, hardware_info: Dict[str, Any]) -> float:
        if not hardware_info:
            return 64.0

        def walk(value: Any):
            if isinstance(value, dict):
                for k, v in value.items():
                    k_low = str(k).lower()
                    if any(key in k_low for key in ["memory", "hbm", "device_memory"]):
                        num = self._parse_memory_number(v)
                        if num:
                            return num
                    found = walk(v)
                    if found:
                        return found
            elif isinstance(value, list):
                for item in value:
                    found = walk(item)
                    if found:
                        return found
            return None

        found = walk(hardware_info)
        return found or 64.0

    def _parse_memory_number(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            if value > 256:
                return float(value) / 1024 / 1024 / 1024
            return float(value)
        if isinstance(value, str):
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
            if match:
                num = float(match.group(1))
                if "mb" in value.lower():
                    return num / 1024
                if "kb" in value.lower():
                    return num / 1024 / 1024
                if "byte" in value.lower() or "bytes" in value.lower():
                    return num / 1024 / 1024 / 1024
                return num
        return None

    def _infer_training_hints(
        self,
        top_operators: List[Dict[str, Any]],
        time_breakdown: Dict[str, float],
        bubble_ratio: float,
    ) -> Dict[str, Any]:
        family_dur = defaultdict(float)
        total_dur = 0.0
        names = []

        for op in top_operators or []:
            name = str(op.get("name", "")).lower()
            dur = float(op.get("dur", 0) or 0)
            total_dur += dur
            names.append(name)
            if self._matches(name, self._ATTENTION_PATTERNS):
                family_dur["attention"] += dur
            if self._matches(name, self._ACTIVATION_AUX_PATTERNS):
                family_dur["activation_aux"] += dur
            if self._matches(name, self._OPTIMIZER_PATTERNS):
                family_dur["optimizer"] += dur
            if self._matches(name, self._MOE_PATTERNS):
                family_dur["moe"] += dur
            if self._matches(name, self._PIPELINE_PATTERNS):
                family_dur["pipeline"] += dur

        total_dur = total_dur or 1.0
        attention_ratio = family_dur["attention"] / total_dur
        activation_aux_ratio = family_dur["activation_aux"] / total_dur
        optimizer_ratio = family_dur["optimizer"] / total_dur
        moe_ratio = family_dur["moe"] / total_dur
        pipeline_ratio = family_dur["pipeline"] / total_dur

        activation_pressure = min(1.0, attention_ratio * 0.75 + activation_aux_ratio * 0.35 + (0.15 if time_breakdown.get("memory", 0) > 0 else 0))
        state_pressure = min(1.0, optimizer_ratio * 1.2 + (0.1 if any("optimizer" in n for n in names) else 0))
        long_context_likely = attention_ratio >= 0.30 or any("flashattention" in n or "attentionscore" in n for n in names)
        moe_likely = moe_ratio >= 0.12 or any("expert" in n or "router" in n or "moe" in n for n in names)
        optimizer_heavy_likely = optimizer_ratio >= 0.10 or any("adam" in n or "optimizer" in n for n in names)
        activation_heavy_likely = activation_pressure >= 0.40

        likely_peak_phase = "unknown"
        if bubble_ratio >= 15 or pipeline_ratio >= 0.10:
            likely_peak_phase = "pipeline_schedule"
        elif optimizer_heavy_likely and state_pressure > activation_pressure:
            likely_peak_phase = "optimizer_step"
        elif activation_heavy_likely:
            likely_peak_phase = "forward_backward"

        return {
            "attention_ratio": round(attention_ratio, 4),
            "activation_aux_ratio": round(activation_aux_ratio, 4),
            "optimizer_ratio": round(optimizer_ratio, 4),
            "moe_ratio": round(moe_ratio, 4),
            "pipeline_ratio": round(pipeline_ratio, 4),
            "activation_pressure_score": round(activation_pressure, 4),
            "state_pressure_score": round(state_pressure, 4),
            "long_context_likely": long_context_likely,
            "moe_likely": moe_likely,
            "optimizer_heavy_likely": optimizer_heavy_likely,
            "activation_heavy_likely": activation_heavy_likely,
            "likely_peak_phase": likely_peak_phase,
        }

    def _matches(self, name: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, name) for pattern in patterns)
