"""
PP Bubble 分析器

分析 Pipeline Parallel 中的 Bubble Time（流水线气泡时间）。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BubbleMetrics:
    """Bubble 分析指标"""
    # PP 配置
    pp_size: int = 1              # Pipeline 并行度
    micro_batches: int = 1        # Micro Batch 数量
    
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
    
    def calculate_ideal_bubble(self) -> float:
        """计算理论 Bubble 比例: (pp_size - 1) / micro_batches"""
        if self.micro_batches == 0:
            return 0.0
        return (self.pp_size - 1) / self.micro_batches
    
    def calculate_bubble_overhead(self) -> float:
        """计算 Bubble 超出理论值的比例"""
        ideal = self.calculate_ideal_bubble()
        if ideal == 0:
            return 0.0
        return max(0, (self.actual_bubble_ratio - ideal) / ideal * 100)
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## PP Bubble 分析",
            "",
            "### 配置信息",
            f"- PP Stage 数量: {self.pp_size}",
            f"- Micro Batch 数量: {self.micro_batches}",
            "",
            "### Bubble 指标",
            f"- 平均 Bubble 时间: {self.avg_bubble_time / 1000:.2f} ms",
            f"- 平均 Stage 时间: {self.avg_stage_time / 1000:.2f} ms",
            f"- 实际 Bubble 比例: {self.actual_bubble_ratio * 100:.1f}%",
            f"- 理论 Bubble 比例: {self.ideal_bubble_ratio * 100:.1f}% (公式: (pp-1)/micro_batches)",
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
    - 1F1B 调度: (pp_size - 1) / micro_batches
    - Interleaved 调度: (pp_size - 1) / (virtual_pp_size * micro_batches)
    """
    
    def __init__(self, pp_size: int = 1, micro_batches: int = 1):
        """
        Args:
            pp_size: Pipeline 并行度
            micro_batches: Micro Batch 数量
        """
        self.pp_size = pp_size
        self.micro_batches = micro_batches
    
    def analyze_from_step_trace(
        self, 
        step_trace_df: pd.DataFrame,
        stage_column: str = "stage",
        bubble_column: str = "bubble",
    ) -> BubbleMetrics:
        """
        从 STEP_TRACE 数据分析 Bubble
        
        Args:
            step_trace_df: 包含 stage, bubble 等列的 DataFrame
            stage_column: Stage 时间列名
            bubble_column: Bubble 时间列名
            
        Returns:
            BubbleMetrics: Bubble 分析结果
        """
        metrics = BubbleMetrics(
            pp_size=self.pp_size,
            micro_batches=self.micro_batches,
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
        
        # 计算理论 Bubble 比例
        metrics.ideal_bubble_ratio = metrics.calculate_ideal_bubble()
        
        # 计算 Bubble 超出理论值
        metrics.bubble_overhead = metrics.calculate_bubble_overhead()
        
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
        
        return metrics
    
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
        
        # 2. 增加 Micro Batch
        if metrics.ideal_bubble_ratio > 0.2 and metrics.micro_batches < 8:
            new_mb = metrics.micro_batches * 2
            new_ideal = (metrics.pp_size - 1) / new_mb
            suggestions.append(
                f"增加 micro_batch 数量（从 {metrics.micro_batches} 到 {new_mb}）"
                f"可将理论 Bubble 从 {metrics.ideal_bubble_ratio*100:.0f}% 降到 {new_ideal*100:.0f}%"
            )
        
        # 3. 使用 Interleaved Schedule
        if metrics.pp_size >= 4 and metrics.actual_bubble_ratio > 0.15:
            suggestions.append(
                "考虑使用 Interleaved PP Schedule（virtual_pp_size=2），"
                "可进一步减少 Bubble Time"
            )
        
        # 4. Stage 负载不均衡
        if metrics.stage_bubbles:
            bubbles = [s["bubble"] for s in metrics.stage_bubbles]
            if len(bubbles) > 1:
                max_bubble = max(bubbles)
                min_bubble = min(bubbles)
                if max_bubble > min_bubble * 1.5:
                    suggestions.append(
                        f"各 Stage Bubble 不均衡（最大 {max_bubble/1000:.2f}ms，"
                        f"最小 {min_bubble/1000:.2f}ms），建议重新划分 PP Stage 以均衡负载"
                    )
        
        return suggestions
