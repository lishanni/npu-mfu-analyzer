"""
数据摘要化模块

将 GB 级 Profiling 数据转换为 KB 级摘要，适合 LLM 分析。
"""

import logging
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
    # 基本信息
    data_path: str = ""
    data_type: str = ""
    framework: str = ""
    rank_count: int = 0
    step_count: int = 0
    
    # 时间指标（平均值，单位 us）
    avg_step_time: float = 0.0
    avg_compute_time: float = 0.0
    avg_comm_time: float = 0.0
    avg_free_time: float = 0.0
    
    # Overlap 指标
    overlap_metrics: OverlapMetrics = field(default_factory=OverlapMetrics)
    
    # Bubble 指标（PP 并行）
    avg_bubble_time: float = 0.0
    bubble_ratio: float = 0.0
    
    # 分类统计
    time_breakdown: Dict[str, float] = field(default_factory=dict)
    top_operators: List[Dict[str, Any]] = field(default_factory=list)
    
    # 原始 Step 数据（采样）
    sample_steps: List[StepMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["overlap_metrics"] = asdict(self.overlap_metrics)
        result["sample_steps"] = [asdict(s) for s in self.sample_steps]
        return result
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式的文本"""
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
        
        if self.top_operators:
            lines.extend([
                "",
                "### Top 10 耗时算子",
            ])
            for i, op in enumerate(self.top_operators[:10], 1):
                lines.append(f"{i}. {op.get('name', 'unknown')}: {op.get('dur', 0) / 1000:.2f} ms")
        
        return "\n".join(lines)
    
    def _ratio(self, part: float, total: float) -> float:
        """计算百分比"""
        return (part / total * 100) if total > 0 else 0


class DataSummarizer:
    """
    数据摘要化器
    
    将 Profiling 数据转换为结构化摘要。
    """
    
    def __init__(self, loader):
        """
        Args:
            loader: ProfilingLoader 实例
        """
        self.loader = loader
    
    def summarize(self, max_sample_steps: int = 10) -> ProfilingSummary:
        """
        生成数据摘要
        
        Args:
            max_sample_steps: 采样的 Step 数量
            
        Returns:
            ProfilingSummary
        """
        info = self.loader.detect()
        
        summary = ProfilingSummary(
            data_path=info.path,
            data_type=info.data_type,
            framework=info.framework,
            rank_count=info.rank_count,
        )
        
        # 获取 Step Trace 数据
        step_trace = self.loader.get_step_trace()
        
        if not step_trace.empty:
            summary.step_count = len(step_trace)
            
            # 计算平均值
            summary.avg_compute_time = step_trace.get("computing", [0]).mean()
            summary.avg_comm_time = step_trace.get("communication", [0]).mean()
            summary.avg_free_time = step_trace.get("free", [0]).mean()
            
            # 计算 step 时间（避免链式赋值，兼容 pandas 3.0 Copy-on-Write）
            # 根据 msprof 的计算逻辑：step_time = Computing + Communication(Not Overlapped) + Free
            # 不能使用 Communication（总通信时间），因为会重复计算重叠的通信时间
            if "computing" in step_trace.columns:
                free_col = step_trace["free"] if "free" in step_trace.columns else 0
                # 优先使用 Communication(Not Overlapped) 字段，避免重复计算重叠时间
                if "communication_not_overlapped" in step_trace.columns:
                    comm_col = step_trace["communication_not_overlapped"]
                elif "communication" in step_trace.columns:
                    # 向后兼容：如果没有非重叠通信时间字段，则使用总通信时间
                    # 但这可能导致 step_time 偏大（重复计算了重叠时间）
                    comm_col = step_trace["communication"]
                else:
                    comm_col = 0
                step_trace = step_trace.assign(
                    step_time=step_trace["computing"] + comm_col + free_col
                )
                summary.avg_step_time = step_trace["step_time"].mean()
            
            # Overlap 指标
            if "communication_not_overlapped" in step_trace.columns:
                summary.overlap_metrics.comm_not_overlapped = step_trace["communication_not_overlapped"].mean()
            if "overlapped" in step_trace.columns:
                summary.overlap_metrics.overlapped_time = step_trace["overlapped"].mean()
            
            total_comm = summary.overlap_metrics.comm_not_overlapped + summary.overlap_metrics.overlapped_time
            if total_comm > 0:
                summary.overlap_metrics.overlap_ratio = (summary.overlap_metrics.overlapped_time / total_comm) * 100
            
            # Bubble 指标
            if "bubble" in step_trace.columns:
                summary.avg_bubble_time = step_trace["bubble"].mean()
                if "stage" in step_trace.columns:
                    avg_stage = step_trace["stage"].mean()
                    if avg_stage > 0:
                        summary.bubble_ratio = (summary.avg_bubble_time / avg_stage) * 100
            
            # 采样 Steps（step 可能为空/NaN，如 CSV 中 Step 列为空）
            sample_indices = step_trace.index[:max_sample_steps]
            for idx in sample_indices:
                row = step_trace.iloc[idx]
                step_val = row.get("step", idx)
                step_int = int(step_val) if step_val == step_val else idx  # NaN != NaN
                summary.sample_steps.append(StepMetrics(
                    step=step_int,
                    computing=float(row.get("computing", 0) or 0),
                    communication=float(row.get("communication", 0)),
                    comm_not_overlapped=float(row.get("communication_not_overlapped", 0)),
                    overlapped=float(row.get("overlapped", 0)),
                    free=float(row.get("free", 0)),
                    stage=float(row.get("stage", 0)),
                    bubble=float(row.get("bubble", 0)),
                ))
        
        # 获取 Timeline 摘要（从 JSON）
        timeline_summary = self.loader.get_timeline_summary()
        if timeline_summary:
            summary.time_breakdown = timeline_summary.get("by_category", {})

        # 获取 Top Kernel 算子（优先从 DB/CSV，过滤掉 Python 栈帧）
        summary.top_operators = self.loader.get_top_kernels(rank=None, top_n=10)

        return summary
