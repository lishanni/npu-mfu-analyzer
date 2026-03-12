"""
通信掩盖率检查技能

分析计算与通信的重叠情况
"""

from typing import Dict, Any, List
import logging

from ..base_skill import (
    BaseSkill,
    SkillMetadata,
    SkillCategory,
    SkillPriority,
    SkillInput,
    SkillOutput,
    SkillResult,
)

logger = logging.getLogger(__name__)


class CheckOverlapRatioSkill(BaseSkill):
    """
    检查通信掩盖率
    
    分析 Computing Stream 和 Communication Stream 的重叠情况
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="check_overlap_ratio",
            display_name="检查通信掩盖率",
            description="分析计算与通信的重叠程度，评估异步通信策略的有效性",
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.CRITICAL,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="total_compute_time_us",
                    type="float",
                    required=True,
                    description="总计算时间（微秒）",
                ),
                SkillInput(
                    name="total_comm_time_us",
                    type="float",
                    required=True,
                    description="总通信时间（微秒）",
                ),
                SkillInput(
                    name="overlapped_time_us",
                    type="float",
                    required=True,
                    description="重叠时间（微秒）",
                ),
                SkillInput(
                    name="free_time_us",
                    type="float",
                    required=False,
                    default=0,
                    description="空闲时间（微秒）",
                ),
            ],
            outputs=[
                SkillOutput(name="overlap_ratio", type="float", description="通信掩盖率 %"),
                SkillOutput(name="comm_not_overlapped_us", type="float", description="未掩盖通信时间"),
                SkillOutput(name="is_optimal", type="bool", description="是否达到最优"),
                SkillOutput(name="potential_speedup", type="float", description="潜在加速比"),
            ],
            tags=["overlap", "communication", "async", "pipeline", "efficiency"],
        )
    
    def execute(
        self,
        total_compute_time_us: float,
        total_comm_time_us: float,
        overlapped_time_us: float,
        free_time_us: float = 0,
        **kwargs,
    ) -> SkillResult:
        """执行掩盖率检查"""
        
        # 计算掩盖率
        overlap_ratio = (overlapped_time_us / total_comm_time_us * 100 
                        if total_comm_time_us > 0 else 100)
        
        # 未掩盖的通信时间
        comm_not_overlapped = total_comm_time_us - overlapped_time_us
        
        # 当前总时间（考虑串行情况）
        # 理想情况：只要计算时间（通信完全掩盖）
        # 实际情况：计算时间 + 未掩盖通信 + 空闲
        current_time = total_compute_time_us + comm_not_overlapped + free_time_us
        ideal_time = total_compute_time_us
        
        # 潜在加速比
        potential_speedup = current_time / ideal_time if ideal_time > 0 else 1.0
        
        # 判断是否最优
        is_optimal = overlap_ratio >= 90 and free_time_us < total_compute_time_us * 0.05
        
        # 生成建议
        suggestions = []
        
        if overlap_ratio < 50:
            suggestions.append("通信掩盖率 < 50%，存在严重的通信瓶颈")
            suggestions.append("建议：")
            suggestions.append("  1. 检查是否使用了异步通信 API")
            suggestions.append("  2. 考虑使用梯度累积减少通信频率")
            suggestions.append("  3. 调整 TP/DP 比例，减少跨节点通信")
        elif overlap_ratio < 70:
            suggestions.append("通信掩盖率 50-70%，有优化空间")
            suggestions.append("建议调整通信与计算的调度顺序")
        elif overlap_ratio < 90:
            suggestions.append("通信掩盖率 70-90%，表现良好")
            if comm_not_overlapped > 0:
                suggestions.append(f"仍有 {comm_not_overlapped/1000:.2f}ms 通信未被掩盖")
        else:
            suggestions.append("通信掩盖率 > 90%，接近最优！")
        
        if free_time_us > total_compute_time_us * 0.1:
            suggestions.append(f"空闲时间较多 ({free_time_us/1000:.2f}ms)，检查 Host-Device 同步")
        
        if potential_speedup > 1.1:
            suggestions.append(f"理论上可通过优化掩盖获得 {(potential_speedup-1)*100:.1f}% 加速")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "overlap_ratio": round(overlap_ratio, 2),
                "comm_not_overlapped_us": round(comm_not_overlapped, 2),
                "comm_not_overlapped_ms": round(comm_not_overlapped / 1000, 2),
                "is_optimal": is_optimal,
                "potential_speedup": round(potential_speedup, 3),
                "total_compute_time_ms": round(total_compute_time_us / 1000, 2),
                "total_comm_time_ms": round(total_comm_time_us / 1000, 2),
                "overlapped_time_ms": round(overlapped_time_us / 1000, 2),
                "free_time_ms": round(free_time_us / 1000, 2),
            },
            summary=f"通信掩盖率 {overlap_ratio:.1f}%，"
                    f"{'已达最优' if is_optimal else f'潜在加速 {(potential_speedup-1)*100:.1f}%'}",
            suggestions=suggestions,
            confidence=0.95,
        )


class VerifyOverlapStrategySkill(BaseSkill):
    """
    验证通信掩盖策略的有效性
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="verify_overlap_strategy",
            display_name="验证掩盖策略",
            description="验证当前的通信掩盖策略是否有效，分析优化方向",
            category=SkillCategory.OPTIMIZATION,
            priority=SkillPriority.NORMAL,
            version="1.0.0",
            inputs=[
                SkillInput(name="overlap_ratio", type="float", required=True,
                          description="当前掩盖率 %"),
                SkillInput(name="comm_pattern", type="str", required=True,
                          description="通信模式: allreduce/p2p/collective"),
                SkillInput(name="parallel_strategy", type="str", required=True,
                          description="并行策略: dp/tp/pp/fsdp"),
                SkillInput(name="gradient_accumulation", type="int", required=False,
                          default=1, description="梯度累积步数"),
            ],
            outputs=[
                SkillOutput(name="strategy_effective", type="bool", description="策略是否有效"),
                SkillOutput(name="recommended_actions", type="list", description="推荐操作"),
            ],
            tags=["overlap", "strategy", "optimization", "parallel"],
        )
    
    def execute(
        self,
        overlap_ratio: float,
        comm_pattern: str,
        parallel_strategy: str,
        gradient_accumulation: int = 1,
        **kwargs,
    ) -> SkillResult:
        """验证掩盖策略"""
        
        strategy_effective = overlap_ratio >= 70
        recommended_actions = []
        
        # 基于并行策略的建议
        strategy_lower = parallel_strategy.lower()
        
        if strategy_lower == "dp" and overlap_ratio < 70:
            recommended_actions.append("DP 模式下掩盖率低，考虑增大 Batch Size 增加计算量")
            if gradient_accumulation == 1:
                recommended_actions.append("启用梯度累积 (gradient_accumulation > 1) 减少通信频率")
        
        elif strategy_lower == "tp" and overlap_ratio < 60:
            recommended_actions.append("TP 模式下掩盖率低是常见问题（通信在关键路径上）")
            recommended_actions.append("考虑减小 TP size，增加 PP 或 DP")
        
        elif strategy_lower == "pp":
            if overlap_ratio < 50:
                recommended_actions.append("PP 模式下掩盖率应较高，检查 P2P 通信实现")
            recommended_actions.append("检查 PP Bubble 时间是否过长")
        
        elif strategy_lower == "fsdp":
            if overlap_ratio < 60:
                recommended_actions.append("FSDP 掩盖率低，检查 prefetch 配置")
                recommended_actions.append("考虑使用 FSDP2 的改进掩盖策略")
        
        # 基于通信模式的建议
        comm_lower = comm_pattern.lower()
        
        if comm_lower == "allreduce" and overlap_ratio < 50:
            recommended_actions.append("AllReduce 通信占比高，考虑：")
            recommended_actions.append("  - 使用 ReduceScatter + AllGather 替代（FSDP 模式）")
            recommended_actions.append("  - 检查是否可以使用 Ring-AllReduce")
        
        confidence = 0.85 if recommended_actions else 0.9
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "strategy_effective": strategy_effective,
                "overlap_ratio": overlap_ratio,
                "comm_pattern": comm_pattern,
                "parallel_strategy": parallel_strategy,
                "gradient_accumulation": gradient_accumulation,
                "recommended_actions": recommended_actions,
            },
            summary=f"掩盖策略{'有效' if strategy_effective else '需优化'}，"
                    f"{len(recommended_actions)} 条建议",
            suggestions=recommended_actions,
            confidence=confidence,
        )
