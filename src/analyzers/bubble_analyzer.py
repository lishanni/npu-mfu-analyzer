"""
PP Bubble 分析器

分析 Pipeline Parallel 中的 Bubble Time（流水线气泡时间）。

支持的调度策略:
- GPipe: 最简单的调度，Bubble = (pp_size - 1) * micro_batch_time
- 1F1B (One Forward One Backward): Bubble = (pp_size - 1) / micro_batches
- Interleaved 1F1B: Bubble = (pp_size - 1) / (virtual_pp_size * micro_batches)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import math

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PPScheduleType(Enum):
    """PP 调度策略类型"""
    GPIPE = "GPipe"
    ONE_F_ONE_B = "1F1B"
    INTERLEAVED_1F1B = "Interleaved_1F1B"
    ZERO_BUBBLE = "ZeroBubble"  # 实验性
    UNKNOWN = "Unknown"


@dataclass
class BubbleBreakdown:
    """Bubble 时间分解"""
    # Warmup 阶段 Bubble (前向传播启动期)
    warmup_bubble_us: float = 0.0
    # Steady-state 阶段（1F1B 稳定期，理论上无 Bubble）
    steady_bubble_us: float = 0.0
    # Cooldown 阶段 Bubble (反向传播收尾期)
    cooldown_bubble_us: float = 0.0
    # 其他 Bubble（如通信等待）
    other_bubble_us: float = 0.0
    
    @property
    def total_us(self) -> float:
        return (self.warmup_bubble_us + self.steady_bubble_us + 
                self.cooldown_bubble_us + self.other_bubble_us)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "warmup": self.warmup_bubble_us,
            "steady": self.steady_bubble_us,
            "cooldown": self.cooldown_bubble_us,
            "other": self.other_bubble_us,
            "total": self.total_us,
        }


@dataclass
class PPScheduleAnalysis:
    """PP 调度策略分析结果"""
    # 识别的调度类型
    schedule_type: PPScheduleType = PPScheduleType.UNKNOWN
    # Virtual PP 大小 (Interleaved 时 > 1)
    virtual_pp_size: int = 1
    # 是否启用 Recomputation (Activation Checkpointing)
    recomputation_enabled: bool = False
    # 检测到的 chunks per stage (Interleaved 特征)
    chunks_per_stage: int = 1
    
    # Bubble 理论值
    theoretical_bubble_ratio: float = 0.0
    # Bubble 实测值
    measured_bubble_ratio: float = 0.0
    # Bubble 效率 (理论/实测，越接近 1 越好)
    bubble_efficiency: float = 0.0
    
    # Bubble 分解
    bubble_breakdown: BubbleBreakdown = field(default_factory=BubbleBreakdown)
    
    # 检测置信度 (0-1)
    detection_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type.value,
            "virtual_pp_size": self.virtual_pp_size,
            "recomputation_enabled": self.recomputation_enabled,
            "chunks_per_stage": self.chunks_per_stage,
            "theoretical_bubble_ratio": self.theoretical_bubble_ratio,
            "measured_bubble_ratio": self.measured_bubble_ratio,
            "bubble_efficiency": self.bubble_efficiency,
            "bubble_breakdown": self.bubble_breakdown.to_dict(),
            "detection_confidence": self.detection_confidence,
        }
    
    def to_prompt_text(self) -> str:
        lines = [
            "## PP 调度策略分析",
            "",
            f"- **调度类型**: {self.schedule_type.value} (置信度: {self.detection_confidence:.0%})",
            f"- **Virtual PP Size**: {self.virtual_pp_size}",
            f"- **Recomputation**: {'启用' if self.recomputation_enabled else '未启用'}",
            "",
            "### Bubble 效率",
            f"- 理论 Bubble 比例: {self.theoretical_bubble_ratio:.1%}",
            f"- 实测 Bubble 比例: {self.measured_bubble_ratio:.1%}",
            f"- Bubble 效率: {self.bubble_efficiency:.1%}",
            "",
            "### Bubble 分解",
            f"- Warmup 阶段: {self.bubble_breakdown.warmup_bubble_us/1000:.2f} ms",
            f"- Steady 阶段: {self.bubble_breakdown.steady_bubble_us/1000:.2f} ms",
            f"- Cooldown 阶段: {self.bubble_breakdown.cooldown_bubble_us/1000:.2f} ms",
            f"- 其他: {self.bubble_breakdown.other_bubble_us/1000:.2f} ms",
        ]
        return "\n".join(lines)


@dataclass
class RecomputationAnalysis:
    """Recomputation (Activation Checkpointing) 分析"""
    # 是否检测到 Recomputation
    detected: bool = False
    # Recomputation 导致的额外计算时间 (us)
    extra_compute_time_us: float = 0.0
    # 节省的 Activation 显存 (估算, MB)
    memory_saved_mb: float = 0.0
    # Recomputation 粒度 (full, selective)
    granularity: str = "unknown"
    # 每个 micro batch 的 recompute 次数
    recompute_count_per_mb: int = 0
    
    def compute_tradeoff_ratio(self, bubble_time_us: float) -> float:
        """
        计算 Recomputation 与 Bubble 的 Trade-off
        
        如果 extra_compute_time > bubble_time，说明可能不值得
        返回值 < 1 表示 Recomputation 带来的计算开销小于潜在的 Bubble 收益
        """
        if bubble_time_us <= 0:
            return float('inf')
        return self.extra_compute_time_us / bubble_time_us
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "extra_compute_time_ms": self.extra_compute_time_us / 1000,
            "memory_saved_mb": self.memory_saved_mb,
            "granularity": self.granularity,
            "recompute_count_per_mb": self.recompute_count_per_mb,
        }


@dataclass
class BubbleMetrics:
    """Bubble 分析指标"""
    # PP 配置
    pp_size: int = 1              # Pipeline 并行度
    micro_batches: int = 1        # Micro Batch 数量
    virtual_pp_size: int = 1      # Virtual PP 大小 (Interleaved)
    
    # 时间指标（平均值，单位 us）
    avg_bubble_time: float = 0.0  # 平均 Bubble 时间
    avg_stage_time: float = 0.0   # 平均 Stage 时间（E2E - Bubble）
    avg_e2e_time: float = 0.0     # 平均端到端时间
    
    # Bubble 比例
    actual_bubble_ratio: float = 0.0    # 实际 Bubble 比例
    ideal_bubble_ratio: float = 0.0     # 理论 Bubble 比例
    bubble_overhead: float = 0.0        # Bubble 超出理论值的比例
    
    # 各 Stage 详情
    stage_bubbles: List[Dict[str, float]] = field(default_factory=list)
    
    # 高级分析结果
    schedule_analysis: Optional[PPScheduleAnalysis] = None
    recomputation_analysis: Optional[RecomputationAnalysis] = None
    
    def calculate_ideal_bubble(self, schedule_type: PPScheduleType = PPScheduleType.ONE_F_ONE_B) -> float:
        """
        计算理论 Bubble 比例
        
        公式:
        - GPipe: (pp_size - 1) / 1  (几乎全是 Bubble)
        - 1F1B: (pp_size - 1) / micro_batches
        - Interleaved 1F1B: (pp_size - 1) / (virtual_pp_size * micro_batches)
        - ZeroBubble: 0 (理论上)
        """
        if self.micro_batches == 0:
            return 0.0
        
        if schedule_type == PPScheduleType.GPIPE:
            # GPipe 的 Bubble 很大，取决于 pp_size
            return (self.pp_size - 1) / self.pp_size
        elif schedule_type == PPScheduleType.INTERLEAVED_1F1B:
            effective_mb = self.virtual_pp_size * self.micro_batches
            return (self.pp_size - 1) / effective_mb if effective_mb > 0 else 0.0
        elif schedule_type == PPScheduleType.ZERO_BUBBLE:
            return 0.0
        else:  # 1F1B
            return (self.pp_size - 1) / self.micro_batches
    
    def calculate_bubble_overhead(self) -> float:
        """计算 Bubble 超出理论值的比例"""
        ideal = self.ideal_bubble_ratio
        if ideal == 0:
            return 0.0 if self.actual_bubble_ratio == 0 else float('inf')
        return max(0, (self.actual_bubble_ratio - ideal) / ideal * 100)
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        schedule_name = "1F1B"
        if self.schedule_analysis:
            schedule_name = self.schedule_analysis.schedule_type.value
        
        lines = [
            "## PP Bubble 分析",
            "",
            "### 配置信息",
            f"- PP Stage 数量: {self.pp_size}",
            f"- Micro Batch 数量: {self.micro_batches}",
            f"- Virtual PP Size: {self.virtual_pp_size}",
            f"- 调度策略: {schedule_name}",
            "",
            "### Bubble 指标",
            f"- 平均 Bubble 时间: {self.avg_bubble_time / 1000:.2f} ms",
            f"- 平均 Stage 时间: {self.avg_stage_time / 1000:.2f} ms",
            f"- 实际 Bubble 比例: {self.actual_bubble_ratio * 100:.1f}%",
            f"- 理论 Bubble 比例: {self.ideal_bubble_ratio * 100:.1f}%",
            f"- Bubble 超出理论值: {self.bubble_overhead:.1f}%",
        ]
        
        if self.stage_bubbles:
            lines.append("")
            lines.append("### 各 Stage Bubble 分布")
            for stage in self.stage_bubbles:
                stage_id = stage.get("stage_id", 0)
                bubble = stage.get("bubble", 0) / 1000  # us -> ms
                stage_time = stage.get("stage_time", 0) / 1000
                lines.append(f"- Stage {stage_id}: Bubble {bubble:.2f}ms / Stage {stage_time:.2f}ms")
        
        # 添加调度分析
        if self.schedule_analysis:
            lines.append("")
            lines.append(self.schedule_analysis.to_prompt_text())
        
        # 添加 Recomputation 分析
        if self.recomputation_analysis and self.recomputation_analysis.detected:
            lines.append("")
            lines.append("### Recomputation 分析")
            lines.append(f"- 粒度: {self.recomputation_analysis.granularity}")
            lines.append(f"- 额外计算时间: {self.recomputation_analysis.extra_compute_time_us/1000:.2f} ms")
            lines.append(f"- 节省显存: {self.recomputation_analysis.memory_saved_mb:.1f} MB (估算)")
            tradeoff = self.recomputation_analysis.compute_tradeoff_ratio(self.avg_bubble_time)
            if tradeoff < float('inf'):
                lines.append(f"- Trade-off 比例: {tradeoff:.2f} (< 1 表示划算)")
        
        return "\n".join(lines)


class BubbleAnalyzer:
    """
    PP Bubble 分析器
    
    分析 Pipeline Parallel 中的 Bubble Time。
    
    Bubble Time 产生原因：
    - PP 中各 Stage 之间存在数据依赖
    - 前向传播时，后面的 Stage 需要等待前面 Stage 的输出
    - 反向传播时，前面的 Stage 需要等待后面 Stage 的梯度
    - Bubble = Receive 操作的等待时间
    
    理论 Bubble 比例：
    - GPipe: (pp_size - 1) / pp_size
    - 1F1B 调度: (pp_size - 1) / micro_batches
    - Interleaved 调度: (pp_size - 1) / (virtual_pp_size * micro_batches)
    - ZeroBubble: 0 (理论上)
    """
    
    def __init__(
        self, 
        pp_size: int = 1, 
        micro_batches: int = 1,
        virtual_pp_size: int = 1,
    ):
        """
        Args:
            pp_size: Pipeline 并行度
            micro_batches: Micro Batch 数量
            virtual_pp_size: Virtual PP 大小 (用于 Interleaved 调度)
        """
        self.pp_size = pp_size
        self.micro_batches = micro_batches
        self.virtual_pp_size = virtual_pp_size
    
    def analyze_from_step_trace(
        self, 
        step_trace_df: pd.DataFrame,
        stage_column: str = "stage",
        bubble_column: str = "bubble",
        operator_df: Optional[pd.DataFrame] = None,
    ) -> BubbleMetrics:
        """
        从 STEP_TRACE 数据分析 Bubble
        
        Args:
            step_trace_df: 包含 stage, bubble 等列的 DataFrame
            stage_column: Stage 时间列名
            bubble_column: Bubble 时间列名
            operator_df: 可选的算子数据，用于检测 Recomputation
            
        Returns:
            BubbleMetrics: Bubble 分析结果
        """
        metrics = BubbleMetrics(
            pp_size=self.pp_size,
            micro_batches=self.micro_batches,
            virtual_pp_size=self.virtual_pp_size,
        )
        
        if step_trace_df.empty:
            return metrics
        
        # 检查必要的列
        if stage_column not in step_trace_df.columns:
            logger.warning(f"Column '{stage_column}' not found in step_trace_df")
            return metrics
        
        if bubble_column not in step_trace_df.columns:
            logger.warning(f"Column '{bubble_column}' not found in step_trace_df")
            return metrics
        
        # 计算平均值
        metrics.avg_stage_time = float(step_trace_df[stage_column].mean())
        metrics.avg_bubble_time = float(step_trace_df[bubble_column].mean())
        
        # 计算 E2E（如果没有单独的 E2E 列，用 stage 作为近似）
        if "e2e" in step_trace_df.columns:
            metrics.avg_e2e_time = float(step_trace_df["e2e"].mean())
        else:
            metrics.avg_e2e_time = metrics.avg_stage_time
        
        # 计算 Bubble 比例
        if metrics.avg_stage_time > 0:
            metrics.actual_bubble_ratio = metrics.avg_bubble_time / metrics.avg_stage_time
        
        # 按 Stage 分组统计（如果有 type/index 列）
        if "type" in step_trace_df.columns and "index" in step_trace_df.columns:
            stage_df = step_trace_df[step_trace_df["type"] == "stage"]
            if not stage_df.empty:
                grouped = stage_df.groupby("index").agg({
                    stage_column: "mean",
                    bubble_column: "mean",
                }).reset_index()
                
                for _, row in grouped.iterrows():
                    metrics.stage_bubbles.append({
                        "stage_id": int(row["index"]),
                        "stage_time": float(row[stage_column]),
                        "bubble": float(row[bubble_column]),
                    })
        
        # 进行调度策略分析
        metrics.schedule_analysis = self.detect_schedule_type(step_trace_df, metrics)
        
        # 更新理论 Bubble 比例（基于检测到的调度类型）
        if metrics.schedule_analysis:
            metrics.ideal_bubble_ratio = metrics.calculate_ideal_bubble(
                metrics.schedule_analysis.schedule_type
            )
        else:
            metrics.ideal_bubble_ratio = metrics.calculate_ideal_bubble()
        
        # 计算 Bubble 超出理论值
        metrics.bubble_overhead = metrics.calculate_bubble_overhead()
        
        # 检测 Recomputation
        if operator_df is not None and not operator_df.empty:
            metrics.recomputation_analysis = self.detect_recomputation(operator_df)
        
        return metrics
    
    def detect_schedule_type(
        self,
        step_trace_df: pd.DataFrame,
        metrics: BubbleMetrics,
    ) -> PPScheduleAnalysis:
        """
        从 Profiling 数据检测 PP 调度策略类型
        
        检测方法:
        1. 通过 Bubble 比例推断
        2. 通过 Forward/Backward 交替模式推断
        3. 通过 chunks_per_stage 推断 (Interleaved 特征)
        """
        analysis = PPScheduleAnalysis(
            virtual_pp_size=self.virtual_pp_size,
            measured_bubble_ratio=metrics.actual_bubble_ratio,
        )
        
        # 计算各调度策略的理论 Bubble
        theoretical_bubbles = {
            PPScheduleType.GPIPE: metrics.calculate_ideal_bubble(PPScheduleType.GPIPE),
            PPScheduleType.ONE_F_ONE_B: metrics.calculate_ideal_bubble(PPScheduleType.ONE_F_ONE_B),
            PPScheduleType.INTERLEAVED_1F1B: metrics.calculate_ideal_bubble(PPScheduleType.INTERLEAVED_1F1B),
            PPScheduleType.ZERO_BUBBLE: 0.0,
        }
        
        # 找到最接近的调度策略
        best_match = PPScheduleType.ONE_F_ONE_B
        min_diff = float('inf')
        
        for schedule_type, theoretical in theoretical_bubbles.items():
            diff = abs(metrics.actual_bubble_ratio - theoretical)
            if diff < min_diff:
                min_diff = diff
                best_match = schedule_type
        
        analysis.schedule_type = best_match
        analysis.theoretical_bubble_ratio = theoretical_bubbles[best_match]
        
        # 计算 Bubble 效率
        if analysis.theoretical_bubble_ratio > 0:
            analysis.bubble_efficiency = min(
                1.0, 
                analysis.theoretical_bubble_ratio / max(analysis.measured_bubble_ratio, 0.001)
            )
        elif analysis.measured_bubble_ratio <= 0.01:
            analysis.bubble_efficiency = 1.0
        else:
            analysis.bubble_efficiency = 0.0
        
        # 计算置信度（基于匹配程度）
        if analysis.theoretical_bubble_ratio > 0:
            relative_diff = min_diff / analysis.theoretical_bubble_ratio
            analysis.detection_confidence = max(0.0, 1.0 - relative_diff)
        else:
            analysis.detection_confidence = 1.0 if metrics.actual_bubble_ratio < 0.05 else 0.5
        
        # 分解 Bubble 时间
        analysis.bubble_breakdown = self._decompose_bubble(step_trace_df, metrics)
        
        return analysis
    
    def _decompose_bubble(
        self,
        step_trace_df: pd.DataFrame,
        metrics: BubbleMetrics,
    ) -> BubbleBreakdown:
        """
        分解 Bubble 时间为 warmup/steady/cooldown 阶段
        
        对于 1F1B 调度:
        - Warmup: 前 (pp_size - 1) 个 micro batch 的前向传播期间
        - Steady: 1F1B 稳定期（理论上无 Bubble）
        - Cooldown: 后 (pp_size - 1) 个 micro batch 的反向传播期间
        """
        breakdown = BubbleBreakdown()
        
        # 简化估算：基于理论公式分配
        total_bubble = metrics.avg_bubble_time
        
        if self.pp_size <= 1 or self.micro_batches <= 0:
            breakdown.other_bubble_us = total_bubble
            return breakdown
        
        # 理论上 warmup 和 cooldown 各占一半
        warmup_steps = self.pp_size - 1
        cooldown_steps = self.pp_size - 1
        total_steps = self.micro_batches
        
        # Warmup 和 Cooldown 的 Bubble 时间估算
        # 每个 warmup step 的平均 Bubble = stage_time / micro_batches * stage_id
        if total_steps > warmup_steps:
            warmup_ratio = warmup_steps / (2 * total_steps)
            cooldown_ratio = warmup_ratio
        else:
            warmup_ratio = 0.5
            cooldown_ratio = 0.5
        
        breakdown.warmup_bubble_us = total_bubble * warmup_ratio
        breakdown.cooldown_bubble_us = total_bubble * cooldown_ratio
        breakdown.steady_bubble_us = max(0, total_bubble - breakdown.warmup_bubble_us - breakdown.cooldown_bubble_us)
        
        return breakdown
    
    def detect_recomputation(
        self,
        operator_df: pd.DataFrame,
    ) -> RecomputationAnalysis:
        """
        从算子数据检测 Recomputation (Activation Checkpointing)
        
        检测特征:
        1. 相同前向算子执行两次
        2. 特定的 checkpoint 相关算子
        3. Forward 计算时间 > 预期（包含 recompute）
        """
        analysis = RecomputationAnalysis()
        
        if operator_df.empty:
            return analysis
        
        # 获取算子名称列
        name_col = None
        for col in ["name", "opName", "op_name"]:
            if col in operator_df.columns:
                name_col = col
                break
        
        if name_col is None:
            return analysis
        
        # 检测特征 1: 查找 checkpoint 相关算子
        checkpoint_keywords = ["checkpoint", "recompute", "activation_checkpoint"]
        op_names = operator_df[name_col].str.lower()
        
        for keyword in checkpoint_keywords:
            if op_names.str.contains(keyword).any():
                analysis.detected = True
                analysis.granularity = "selective" if "selective" in keyword else "full"
                break
        
        # 检测特征 2: 查找重复的前向算子
        if not analysis.detected:
            # 统计每个算子的执行次数
            op_counts = operator_df[name_col].value_counts()
            
            # 如果有大量重复执行的计算算子，可能是 recomputation
            matmul_ops = op_counts[op_counts.index.str.contains("MatMul|matmul", case=False)]
            if not matmul_ops.empty:
                avg_count = matmul_ops.mean()
                if avg_count > 1.5:  # 平均执行次数超过 1.5 次
                    analysis.detected = True
                    analysis.granularity = "full"
                    analysis.recompute_count_per_mb = int(avg_count)
        
        # 估算额外计算时间
        if analysis.detected:
            dur_col = None
            for col in ["dur", "duration", "time"]:
                if col in operator_df.columns:
                    dur_col = col
                    break
            
            if dur_col:
                total_compute = operator_df[dur_col].sum()
                # 假设 recomputation 增加了约 33% 的计算时间
                analysis.extra_compute_time_us = total_compute * 0.25
                
                # 估算节省的显存（基于层数和 hidden_size 的粗略估算）
                # 实际需要从模型配置获取
                analysis.memory_saved_mb = self.pp_size * 100  # 粗略估算
        
        return analysis
    
    def analyze_from_summary(
        self, 
        summary_data: Dict[str, Any],
    ) -> BubbleMetrics:
        """
        从摘要数据分析 Bubble
        
        Args:
            summary_data: 包含 avg_bubble_time, bubble_ratio 等的字典
            
        Returns:
            BubbleMetrics
        """
        metrics = BubbleMetrics(
            pp_size=self.pp_size,
            micro_batches=self.micro_batches,
        )
        
        metrics.avg_bubble_time = float(summary_data.get("avg_bubble_time", 0))
        metrics.actual_bubble_ratio = float(summary_data.get("bubble_ratio", 0)) / 100  # % -> 小数
        
        # 从 step_time 估算 stage_time
        if "avg_step_time" in summary_data:
            metrics.avg_stage_time = float(summary_data["avg_step_time"])
            if metrics.avg_stage_time > 0 and metrics.actual_bubble_ratio == 0:
                metrics.actual_bubble_ratio = metrics.avg_bubble_time / metrics.avg_stage_time
        
        metrics.ideal_bubble_ratio = metrics.calculate_ideal_bubble()
        metrics.bubble_overhead = metrics.calculate_bubble_overhead()
        
        return metrics
    
    def suggest_optimization(self, metrics: BubbleMetrics) -> List[str]:
        """
        根据 Bubble 分析结果给出优化建议
        
        Args:
            metrics: BubbleMetrics
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        # 1. Bubble 比例过高
        if metrics.actual_bubble_ratio > metrics.ideal_bubble_ratio * 1.2:
            overhead_pct = metrics.bubble_overhead
            suggestions.append(
                f"Bubble 比例 ({metrics.actual_bubble_ratio*100:.1f}%) 超出理论值 "
                f"({metrics.ideal_bubble_ratio*100:.1f}%) {overhead_pct:.0f}%，"
                "建议检查是否有通信阻塞或负载不均衡"
            )
        
        # 2. 根据检测到的调度类型给出建议
        if metrics.schedule_analysis:
            schedule = metrics.schedule_analysis
            
            # 如果检测到 GPipe，建议升级到 1F1B
            if schedule.schedule_type == PPScheduleType.GPIPE:
                new_ideal = metrics.calculate_ideal_bubble(PPScheduleType.ONE_F_ONE_B)
                improvement = (metrics.actual_bubble_ratio - new_ideal) / metrics.actual_bubble_ratio * 100
                suggestions.append(
                    f"当前使用 GPipe 调度，建议升级到 1F1B 调度，"
                    f"可将 Bubble 从 {metrics.actual_bubble_ratio*100:.1f}% 降至 {new_ideal*100:.1f}% "
                    f"(预计提升 {improvement:.0f}%)"
                )
            
            # 如果使用 1F1B 且 Bubble 仍然较高，建议 Interleaved
            elif schedule.schedule_type == PPScheduleType.ONE_F_ONE_B:
                if metrics.pp_size >= 4 and metrics.actual_bubble_ratio > 0.1:
                    suggested_vpp = min(4, metrics.pp_size // 2)
                    new_ideal = (metrics.pp_size - 1) / (suggested_vpp * metrics.micro_batches)
                    suggestions.append(
                        f"建议使用 Interleaved 1F1B 调度（virtual_pp_size={suggested_vpp}），"
                        f"可将理论 Bubble 从 {metrics.ideal_bubble_ratio*100:.1f}% 降至 {new_ideal*100:.1f}%"
                    )
            
            # 如果已经是 Interleaved 但效率不高
            elif schedule.schedule_type == PPScheduleType.INTERLEAVED_1F1B:
                if schedule.bubble_efficiency < 0.8:
                    suggestions.append(
                        f"Interleaved 调度的 Bubble 效率较低 ({schedule.bubble_efficiency:.0%})，"
                        "可能存在 Stage 间负载不均衡或通信瓶颈"
                    )
        
        # 3. 增加 Micro Batch
        if metrics.ideal_bubble_ratio > 0.15 and metrics.micro_batches < 16:
            new_mb = min(metrics.micro_batches * 2, 32)
            if metrics.schedule_analysis and metrics.schedule_analysis.schedule_type == PPScheduleType.INTERLEAVED_1F1B:
                new_ideal = (metrics.pp_size - 1) / (metrics.virtual_pp_size * new_mb)
            else:
                new_ideal = (metrics.pp_size - 1) / new_mb
            
            if new_ideal < metrics.ideal_bubble_ratio * 0.7:  # 至少 30% 的提升
                suggestions.append(
                    f"增加 micro_batch 数量（从 {metrics.micro_batches} 到 {new_mb}）"
                    f"可将理论 Bubble 从 {metrics.ideal_bubble_ratio*100:.0f}% 降到 {new_ideal*100:.0f}%"
                )
        
        # 4. Stage 负载不均衡
        if metrics.stage_bubbles:
            bubbles = [s["bubble"] for s in metrics.stage_bubbles]
            if len(bubbles) > 1:
                max_bubble = max(bubbles)
                min_bubble = min(bubbles) if min(bubbles) > 0 else 1
                imbalance_ratio = max_bubble / min_bubble
                if imbalance_ratio > 1.5:
                    suggestions.append(
                        f"各 Stage Bubble 不均衡（比例 {imbalance_ratio:.1f}x，"
                        f"最大 {max_bubble/1000:.2f}ms，最小 {min_bubble/1000:.2f}ms），"
                        "建议重新划分 PP Stage 以均衡负载"
                    )
        
        # 5. Recomputation Trade-off 分析
        if metrics.recomputation_analysis and metrics.recomputation_analysis.detected:
            recomp = metrics.recomputation_analysis
            tradeoff = recomp.compute_tradeoff_ratio(metrics.avg_bubble_time)
            
            if tradeoff > 1.5:
                suggestions.append(
                    f"Recomputation 带来的额外计算时间 ({recomp.extra_compute_time_us/1000:.2f}ms) "
                    f"超过 Bubble 时间的 {tradeoff:.1f} 倍，建议评估是否需要禁用 Recomputation"
                )
            elif tradeoff < 0.5 and metrics.actual_bubble_ratio > 0.1:
                suggestions.append(
                    f"Recomputation 开销较小 (Trade-off={tradeoff:.2f})，"
                    "可以考虑更激进的 Checkpointing 策略以减少显存占用"
                )
        
        # 6. Bubble 分解分析
        if metrics.schedule_analysis and metrics.schedule_analysis.bubble_breakdown:
            breakdown = metrics.schedule_analysis.bubble_breakdown
            total = breakdown.total_us
            if total > 0:
                # 如果 steady 阶段有异常 Bubble
                if breakdown.steady_bubble_us > total * 0.2:
                    suggestions.append(
                        f"Steady 阶段存在异常 Bubble ({breakdown.steady_bubble_us/1000:.2f}ms，"
                        f"占比 {breakdown.steady_bubble_us/total*100:.0f}%)，"
                        "可能是由于 P2P 通信延迟或计算负载不均衡导致"
                    )
        
        return suggestions
    
    def compare_schedule_strategies(self) -> Dict[str, Dict[str, float]]:
        """
        比较不同调度策略的理论性能
        
        Returns:
            各调度策略的理论 Bubble 比例和效率提升
        """
        base_bubble = (self.pp_size - 1) / self.micro_batches  # 1F1B 作为基准
        
        strategies = {
            "GPipe": {
                "bubble_ratio": (self.pp_size - 1) / self.pp_size,
                "relative_to_1f1b": None,
            },
            "1F1B": {
                "bubble_ratio": base_bubble,
                "relative_to_1f1b": 1.0,
            },
        }
        
        # 计算不同 VPP 配置的 Interleaved
        for vpp in [2, 4, 8]:
            if vpp <= self.pp_size:
                bubble = (self.pp_size - 1) / (vpp * self.micro_batches)
                strategies[f"Interleaved_VPP{vpp}"] = {
                    "bubble_ratio": bubble,
                    "relative_to_1f1b": bubble / base_bubble if base_bubble > 0 else 0,
                }
        
        strategies["ZeroBubble"] = {
            "bubble_ratio": 0.0,
            "relative_to_1f1b": 0.0,
        }
        
        # 计算相对 GPipe 的改进
        gpipe_bubble = strategies["GPipe"]["bubble_ratio"]
        for name, data in strategies.items():
            if gpipe_bubble > 0:
                data["improvement_vs_gpipe"] = (gpipe_bubble - data["bubble_ratio"]) / gpipe_bubble
            else:
                data["improvement_vs_gpipe"] = 0.0
        
        return strategies
    
    def estimate_optimal_config(
        self,
        target_bubble_ratio: float = 0.05,
        max_micro_batches: int = 64,
        max_virtual_pp: int = 8,
    ) -> Dict[str, Any]:
        """
        估算达到目标 Bubble 比例所需的配置
        
        Args:
            target_bubble_ratio: 目标 Bubble 比例
            max_micro_batches: 最大 Micro Batch 数量
            max_virtual_pp: 最大 Virtual PP Size
            
        Returns:
            推荐配置
        """
        recommendations = []
        
        # 1. 计算 1F1B 所需的 micro_batches
        required_mb_1f1b = (self.pp_size - 1) / target_bubble_ratio if target_bubble_ratio > 0 else float('inf')
        if required_mb_1f1b <= max_micro_batches:
            recommendations.append({
                "strategy": "1F1B",
                "micro_batches": math.ceil(required_mb_1f1b),
                "virtual_pp_size": 1,
                "achievable_bubble": (self.pp_size - 1) / math.ceil(required_mb_1f1b),
            })
        
        # 2. 计算 Interleaved 所需的配置
        for vpp in range(2, min(max_virtual_pp + 1, self.pp_size + 1)):
            required_mb = (self.pp_size - 1) / (vpp * target_bubble_ratio) if target_bubble_ratio > 0 else float('inf')
            if required_mb <= max_micro_batches:
                recommendations.append({
                    "strategy": "Interleaved_1F1B",
                    "micro_batches": math.ceil(required_mb),
                    "virtual_pp_size": vpp,
                    "achievable_bubble": (self.pp_size - 1) / (vpp * math.ceil(required_mb)),
                })
        
        # 按 micro_batches 排序（更小更好，因为显存占用更低）
        recommendations.sort(key=lambda x: (x["micro_batches"], -x["virtual_pp_size"]))
        
        return {
            "target_bubble_ratio": target_bubble_ratio,
            "pp_size": self.pp_size,
            "recommendations": recommendations[:5],  # Top 5 配置
        }
