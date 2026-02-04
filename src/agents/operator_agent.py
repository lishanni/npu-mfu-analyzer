"""
Operator Agent

分析算子性能，计算 MFU，识别低效算子，检测融合机会。
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import pandas as pd

from src.agents.base_agent import BaseAgent, AnalysisResult
from src.llm.llm_interface import LLMInterface
from src.llm.prompts import MFU_ANALYSIS_SYSTEM
from src.analyzers.mfu_calculator import MFUCalculator, MFUMetrics, ChipInfo
from src.agents.fusion_rules import (
    FusionPattern,
    FusionOpportunity,
    FUSION_PATTERNS,
    ASCEND_FUSED_OPERATORS,
    find_ascend_fused_op,
)

logger = logging.getLogger(__name__)


class FusionAnalyzer:
    """
    算子融合机会分析器

    检测 Profiling 数据中的融合机会，包括：
    1. 昇腾已有的融合算子（可直接替换）
    2. 可以进一步融合的算子序列（需要自定义实现）
    3. 自定义融合建议（可用 Triton 等实现）

    关键特性：
    - 全局分析：不只关注 Top 耗时算子，分析所有算子
    - 端到端加速：考虑算子在全图中的耗时占比，计算实际端到端训练加速
    """

    def __init__(self):
        self.ascend_fused_ops = ASCEND_FUSED_OPERATORS
        self.fusion_patterns = FUSION_PATTERNS
        self._total_duration_us = 0.0  # 总执行时间（微秒）

    def detect_opportunities(
        self,
        all_operators: List[Dict[str, Any]],
        top_operators: Optional[List[Dict[str, Any]]] = None,
        timeline_data: Optional[List[Dict]] = None
    ) -> List[FusionOpportunity]:
        """
        检测融合机会

        Args:
            all_operators: 所有算子列表（用于全局分析和计算总耗时）
            top_operators: Top 耗时算子列表（可选，如果未提供则从 all_operators 取前 N 个）
            timeline_data: Timeline 事件列表（用于分析算子序列）

        Returns:
            融合机会列表，按端到端加速效果排序
        """
        opportunities = []

        # 计算总执行时间
        self._total_duration_us = self._calculate_total_duration(all_operators)

        # 如果没有提供 top_operators，从 all_operators 中提取
        if top_operators is None:
            top_operators = sorted(all_operators, key=lambda x: x.get("dur", 0), reverse=True)[:50]

        # 1. 全局分析：检测未使用昇腾融合算子的地方
        opportunities.extend(self._detect_missing_fusion_ops(all_operators))

        # 2. 分析算子序列，检测可进一步融合的模式
        if timeline_data:
            opportunities.extend(self._detect_sequence_fusion_patterns(all_operators, timeline_data))

        # 3. 基于算子类型推断融合机会（全局）
        opportunities.extend(self._detect_type_based_fusions(all_operators))

        # 4. 计算端到端加速效果并按端到端收益排序
        for opp in opportunities:
            opp = self._calculate_end_to_end_speedup(opp)

        # 按端到端加速效果排序（而非算子级别加速）
        opportunities.sort(key=lambda x: x.end_to_end_speedup, reverse=True)

        return opportunities

    def _calculate_total_duration(self, operators: List[Dict[str, Any]]) -> float:
        """计算所有算子的总执行时间（微秒）"""
        return sum(op.get("dur", 0) for op in operators)

    def _calculate_end_to_end_speedup(self, opp: FusionOpportunity) -> FusionOpportunity:
        """
        计算融合机会的端到端加速效果

        端到端加速计算公式：
        设原总时间为 T，涉及算子时间为 t_op，算子加速比为 s
        新总时间 T_new = T - t_op + t_op/s
        端到端加速 = T / T_new

        Args:
            opp: 融合机会

        Returns:
            更新后的融合机会（包含端到端加速信息）
        """
        if self._total_duration_us <= 0:
            opp.end_to_end_speedup = 1.0
            opp.time_proportion = 0.0
            return opp

        # 计算涉及算子的总耗时
        total_op_dur = sum(op.get("dur", 0) for op in opp.current_ops)
        opp.total_op_duration_us = total_op_dur

        # 计算时间占比
        opp.time_proportion = total_op_dur / self._total_duration_us

        # 计算端到端加速
        # T_new = T - t_op + t_op/s
        # speedup = T / T_new
        if opp.estimated_speedup > 1.0 and total_op_dur > 0:
            new_total = self._total_duration_us - total_op_dur + (total_op_dur / opp.estimated_speedup)
            opp.end_to_end_speedup = self._total_duration_us / new_total if new_total > 0 else 1.0
        else:
            opp.end_to_end_speedup = 1.0

        return opp

    def _get_operators_by_type(
        self,
        all_operators: List[Dict[str, Any]],
        keywords: List[str],
        max_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        从所有算子中按关键词查找（全局搜索）

        Args:
            all_operators: 所有算子列表
            keywords: 关键词列表（简单子串匹配）
            max_count: 最大返回数量（按耗时排序后取前 N 个）

        Returns:
            匹配的算子列表
        """
        matched = self._find_ops_by_keywords(all_operators, keywords)
        # 按耗时排序
        matched.sort(key=lambda x: x.get("dur", 0), reverse=True)
        if max_count:
            return matched[:max_count]
        return matched

    def _detect_missing_fusion_ops(self, all_operators: List[Dict]) -> List[FusionOpportunity]:
        """
        检测未使用昇腾融合算子的场景（全局分析）

        Args:
            all_operators: 所有算子列表

        Returns:
            融合机会列表
        """
        opportunities = []

        if not all_operators:
            return opportunities

        op_names = [op.get("name", "") for op in all_operators]

        # 检测 Attention 算子但未使用 FlashAttention
        has_attention = any(
            self._matches_keywords(name, ["attention", "attn"])
            for name in op_names
        )
        has_flash = any(
            self._matches_keywords(name, ["flashattention", "flash_attention"])
            for name in op_names
        )

        if has_attention and not has_flash:
            # 全局查找所有 Attention 相关算子
            attn_ops = self._find_ops_by_keywords(all_operators, ["attention", "attn"])
            if attn_ops:
                opportunities.append(FusionOpportunity(
                    opportunity_type="replace",
                    name="使用 FlashAttention 替换普通 Attention",
                    description=f"将标准 Attention 实现替换为 FlashAttention，涉及 {len(attn_ops)} 个 Attention 算子",
                    current_ops=attn_ops[:20],
                    estimated_speedup=5.0,
                    memory_saving=0.6,
                    implementation="昇腾原生支持，使用 aclnnFlashAttentionScore 直接替换",
                    complexity="低",
                    ascend_op="aclnnFlashAttentionScore",
                ))

        # 检测 MatMul + Bias + Activation 序列（全局统计）
        # 获取所有 MatMul 算子
        all_matmul = self._find_ops_by_keywords(all_operators, ["matmul"])
        # 获取所有 Bias/Add 算子（使用简单关键词）
        all_bias = self._find_ops_by_keywords(all_operators, ["bias", "biasadd"])
        # 获取所有激活函数算子
        all_act = self._find_ops_by_keywords(all_operators, ["gelu", "relu", "silu", "sigmoid"])

        # 如果有足够的 MatMul+Bias+Act 组合，建议融合
        matmul_count = len(all_matmul)
        fusion_count = min(matmul_count, len(all_bias), len(all_act))

        if fusion_count >= 2:  # 至少 2 层可以融合
            # 选取耗时最大的几个算子作为示例
            sample_ops = (all_matmul[:2] + all_bias[:1] + all_act[:1])[:4]
            opportunities.append(FusionOpportunity(
                opportunity_type="fuse",
                name=f"MatMul+Bias+Activation 融合（约 {fusion_count} 处）",
                description=f"将矩阵乘法、偏置加法和激活函数融合，预计可融合约 {fusion_count} 处",
                current_ops=sample_ops,
                estimated_speedup=1.3,
                memory_saving=0.4,
                implementation="使用昇腾 aclnnFusedMatMulBiasAct 或 Triton 自定义实现",
                complexity="中等",
                ascend_op="aclnnFusedMatMulBiasAct",
            ))

        # 检测 LayerNorm/RMSNorm + Add 残差连接（全局统计）
        all_norm = self._find_ops_by_keywords(all_operators, ["layernorm", "rmsnorm", "norm"])
        all_add = self._find_ops_by_keywords(all_operators, ["add"])

        norm_add_count = min(len(all_norm), len(all_add))
        if norm_add_count >= 2:
            sample_ops = (all_norm[:2] + all_add[:1])[:3]
            opportunities.append(FusionOpportunity(
                opportunity_type="fuse",
                name=f"Norm + Residual Add 融合（约 {norm_add_count} 处）",
                description=f"将归一化和残差连接融合，预计可融合约 {norm_add_count} 处",
                current_ops=sample_ops,
                estimated_speedup=1.2,
                memory_saving=0.3,
                implementation="使用昇腾 aclnnAddRmsNorm 或 aclnnFusedLayerNorm",
                complexity="低",
                ascend_op="aclnnAddRmsNorm",
            ))

        # 检测 QKV 投影分离（全局统计）- 使用简单关键词
        qkv_ops = self._find_ops_by_keywords(all_operators, [
            "query", "_q", "q_proj",
            "key", "_k", "k_proj",
            "value", "_v", "v_proj"
        ])

        # 简单估算：每 3 个 QKV 投影可以融合成 1 个
        qkv_fusion_count = len(qkv_ops) // 3
        if qkv_fusion_count >= 1:
            opportunities.append(FusionOpportunity(
                opportunity_type="fuse",
                name=f"QKV Projection 融合（约 {qkv_fusion_count} 处）",
                description=f"将 Q、K、V 三个独立投影合并，预计可融合约 {qkv_fusion_count} 处",
                current_ops=qkv_ops[:3],
                estimated_speedup=1.5,
                memory_saving=0.35,
                implementation="使用昇腾 aclnnFusedQKVProjection 或合并权重矩阵",
                complexity="低",
                ascend_op="aclnnFusedQKVProjection",
            ))

        return opportunities

    def _detect_sequence_fusion_patterns(
        self,
        all_operators: List[Dict],
        timeline: List[Dict]
    ) -> List[FusionOpportunity]:
        """
        分析 Timeline 中的算子序列，检测融合模式（全局分析）

        Args:
            all_operators: 所有算子列表
            timeline: Timeline 事件列表

        Returns:
            融合机会列表
        """
        opportunities = []

        # 简化实现：基于算子列表检测序列模式
        # 实际项目中可以基于 Timeline 时间戳分析相邻算子

        # 检测逐元素操作链（这部分已在 _detect_type_based_fusions 中处理）
        # 这里可以添加更复杂的序列检测逻辑

        return opportunities

    def _detect_type_based_fusions(self, all_operators: List[Dict]) -> List[FusionOpportunity]:
        """
        基于算子类型推断融合机会（全局分析）

        Args:
            all_operators: 所有算子列表

        Returns:
            融合机会列表
        """
        opportunities = []

        # 检测 MoE 相关算子
        moe_ops = self._get_operators_by_type(all_operators, [
            "moe", "expert", "routing", "topk", "grouped"
        ], max_count=20)

        if moe_ops:
            moe_dur = sum(op.get("dur", 0) for op in moe_ops)
            moe_ratio = moe_dur / self._total_duration_us if self._total_duration_us > 0 else 0
            opportunities.append(FusionOpportunity(
                opportunity_type="fuse",
                name=f"MoE 专家计算融合（{len(moe_ops)} 个算子，耗时占比 {moe_ratio:.1%}）",
                description=f"使用 GroupedMatMul 融合 MoE 专家计算，涉及 {len(moe_ops)} 个相关算子",
                current_ops=moe_ops[:5],
                estimated_speedup=1.4,
                memory_saving=0.4,
                implementation="使用昇腾 aclnnGroupedMatmulV4",
                complexity="低",
                ascend_op="aclnnGroupedMatmulV4",
            ))

        # 检测 reshape/transpose 后接计算（全局统计）
        reshape_ops = self._get_operators_by_type(all_operators, [
            "reshape", "transpose", "permute", "view"
        ])

        reshape_count = len(reshape_ops)
        if reshape_count >= 5:
            reshape_dur = sum(op.get("dur", 0) for op in reshape_ops)
            reshape_ratio = reshape_dur / self._total_duration_us if self._total_duration_us > 0 else 0
            opportunities.append(FusionOpportunity(
                opportunity_type="optimize",
                name=f"减少形状变换操作（{reshape_count} 个，耗时占比 {reshape_ratio:.1%}）",
                description=f"检测到 {reshape_count} 个 reshape/transpose 操作，总耗时占比 {reshape_ratio:.1%}，建议优化内存布局或融合到相邻算子",
                current_ops=reshape_ops[:5],
                estimated_speedup=1.1,
                memory_saving=0.15,
                implementation="优化张量内存布局，避免不必要的形状变换",
                complexity="中等",
            ))

        # 检测 Cast 操作（全局统计）
        cast_ops = self._get_operators_by_type(all_operators, ["cast"])
        cast_count = len(cast_ops)
        if cast_count >= 3:
            cast_dur = sum(op.get("dur", 0) for op in cast_ops)
            cast_ratio = cast_dur / self._total_duration_us if self._total_duration_us > 0 else 0
            opportunities.append(FusionOpportunity(
                opportunity_type="optimize",
                name=f"减少数据类型转换（{cast_count} 个，耗时占比 {cast_ratio:.1%}）",
                description=f"检测到 {cast_count} 个 Cast 操作，总耗时占比 {cast_ratio:.1%}，建议统一数据类型或融合到计算中",
                current_ops=cast_ops[:5],
                estimated_speedup=1.05,
                memory_saving=0.1,
                implementation="统一使用 FP16/BF16，将 Cast 融合到算子中",
                complexity="简单",
            ))

        # 检测逐元素操作链（全局统计）
        element_ops = self._get_operators_by_type(all_operators, [
            "dropout", "scale"
        ])
        element_count = len(element_ops)
        if element_count >= 10:
            element_dur = sum(op.get("dur", 0) for op in element_ops)
            element_ratio = element_dur / self._total_duration_us if self._total_duration_us > 0 else 0
            opportunities.append(FusionOpportunity(
                opportunity_type="fuse",
                name=f"逐元素操作融合（{element_count} 个，耗时占比 {element_ratio:.1%}）",
                description=f"检测到 {element_count} 个逐元素操作，可融合为更少的 Kernel",
                current_ops=element_ops[:5],
                estimated_speedup=1.15,
                memory_saving=0.25,
                implementation="使用 Triton 或昇腾融合算子",
                complexity="简单",
                ascend_op="aclnnFusedDropoutAdd",
            ))

        return opportunities

    def _find_ops_by_keywords(
        self,
        operators: List[Dict],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        在算子列表中查找包含特定关键词的算子

        Args:
            operators: 算子列表
            keywords: 关键词列表（支持子串匹配，不区分大小写）

        Returns:
            匹配的算子列表
        """
        matched = []
        # 不使用 re.escape，直接用 | 连接作为正则 OR 模式
        # 但需要转义正则特殊字符，避免意外匹配
        escaped_keywords = [re.escape(k) for k in keywords]
        pattern = "|".join(escaped_keywords)

        for op in operators:
            name = op.get("name", "")
            if re.search(pattern, name, re.IGNORECASE):
                matched.append(op)

        return matched

    def _matches_keywords(self, name: str, keywords: List[str]) -> bool:
        """检查名称是否匹配任意关键词"""
        escaped_keywords = [re.escape(k) for k in keywords]
        pattern = "|".join(escaped_keywords)
        return bool(re.search(pattern, name, re.IGNORECASE))


@dataclass
class OperatorAnalysisData:
    """算子分析数据"""
    # MFU 指标
    mfu_metrics: Optional[MFUMetrics] = None

    # 算子统计
    total_operators: int = 0
    top_operators: List[Dict[str, Any]] = field(default_factory=list)
    low_efficiency_operators: List[Dict[str, Any]] = field(default_factory=list)

    # 融合机会分析
    fusion_opportunities: List[FusionOpportunity] = field(default_factory=list)

    # 芯片信息
    chip_name: str = ""
    peak_flops_tflops: float = 0.0

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 算子分析数据摘要",
            "",
            f"### 芯片信息",
            f"- 芯片型号: {self.chip_name or '未知'}",
            f"- 理论峰值: {self.peak_flops_tflops:.1f} TFLOPS",
            "",
        ]

        if self.mfu_metrics:
            lines.append(self.mfu_metrics.to_prompt_text())
        else:
            lines.append("### MFU 分析")
            lines.append("- 无 MFU 数据（需要算子形状信息）")

        if self.top_operators:
            lines.append("")
            lines.append("### Top 10 耗时算子")
            for i, op in enumerate(self.top_operators[:10], 1):
                name = op.get("name", "unknown")
                dur = op.get("dur", 0) / 1000  # us -> ms
                lines.append(f"{i}. {name}: {dur:.2f} ms")

        if self.low_efficiency_operators:
            lines.append("")
            lines.append("### 低效算子（需优化）")
            for op in self.low_efficiency_operators[:5]:
                name = op.get("name", "unknown")
                mfu = op.get("mfu", 0) * 100
                lines.append(f"- {name}: MFU={mfu:.1f}%")

        # 融合机会分析
        if self.fusion_opportunities:
            lines.extend([
                "",
                "### 算子融合机会分析",
                f"- 发现 {len(self.fusion_opportunities)} 个融合机会",
                "",
                "### 推荐的融合优化",
            ])
            for i, opp in enumerate(self.fusion_opportunities[:5], 1):
                lines.append(opp.to_prompt_text())
                lines.append("")

        return "\n".join(lines)


class OperatorAgent(BaseAgent):
    """
    Operator Agent

    功能：
    1. MFU 计算（Model FLOPS Utilization）
    2. 算子耗时分析
    3. 低效算子识别
    4. 算子融合机会检测
    5. 优化建议生成
    """

    PROMPT_TEMPLATE = """
你是昇腾 NPU 算子优化专家。分析以下算子性能数据：

{data_summary}

## 分析任务
1. **MFU 评估**：
   - MFU > 50%：算子效率良好
   - MFU 30-50%：有优化空间
   - MFU < 30%：需要重点优化

2. **瓶颈识别**：
   - 哪些算子占用时间最多？
   - 哪些算子 MFU 最低？
   - 是否有冗余计算？

3. **融合机会分析**：
   - 检查是否有可替换的昇腾融合算子
   - 识别可进一步融合的算子序列
   - 评估融合的性能收益和实现复杂度

4. **优化建议**：
   - 算子融合（Operator Fusion）
   - 数据类型优化（FP16/BF16/INT8）
   - FlashAttention 替换普通 Attention
   - 减少不必要的 reshape/transpose

请给出详细的分析结论和具体优化建议。对于融合机会，请提供：
1. 是否有昇腾原生算子可直接替换
2. 自定义融合的实现难度
3. 预期的性能提升和内存节省
"""

    def __init__(
        self,
        llm: LLMInterface,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="OperatorAgent",
            llm=llm,
            system_prompt=MFU_ANALYSIS_SYSTEM,
            config=config
        )
        self._mfu_calculator: Optional[MFUCalculator] = None
        self._fusion_analyzer = FusionAnalyzer()

    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析算子数据

        Args:
            data: 包含以下可选字段：
                - operators_df: 算子数据 DataFrame（所有算子，用于全局融合分析）
                - top_operators: Top 耗时算子列表
                - profiling_path: Profiling 数据路径（用于加载芯片信息）
                - chip_info: ChipInfo 对象
                - timeline_events: Timeline 事件列表（用于融合分析）

        Returns:
            AnalysisResult
        """
        try:
            # 1. 准备分析数据
            analysis_data = self._prepare_analysis_data(data)

            # 2. 计算 MFU（如果有算子数据）
            if "operators_df" in data:
                analysis_data.mfu_metrics = self._calculate_mfu(
                    data["operators_df"],
                    data.get("chip_info")
                )

            # 3. 检测融合机会（新增：支持全局分析）
            # 优先使用 operators_df 进行全局分析，否则使用 top_operators
            all_operators = None
            if "operators_df" in data:
                df = data["operators_df"]
                # 将 DataFrame 转换为算子列表
                all_operators = df.to_dict("records") if not df.empty else []
            elif "all_operators" in data:
                all_operators = data["all_operators"]

            if all_operators or "top_operators" in data:
                analysis_data.fusion_opportunities = self._fusion_analyzer.detect_opportunities(
                    all_operators=all_operators if all_operators else analysis_data.top_operators,
                    top_operators=analysis_data.top_operators if all_operators else None,
                    timeline_data=data.get("timeline_events")
                )

            # 4. 生成 Prompt 并调用 LLM
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                data_summary=analysis_data.to_prompt_text()
            )
            response = await self.call_llm(prompt)

            # 5. 构建结果
            mfu_value = analysis_data.mfu_metrics.overall_mfu if analysis_data.mfu_metrics else 0

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=(
                    f"MFU: {mfu_value*100:.1f}%, "
                    f"分析 {analysis_data.total_operators} 个算子, "
                    f"发现 {len(analysis_data.fusion_opportunities)} 个融合机会"
                ),
                details={
                    "overall_mfu": mfu_value,
                    "matmul_mfu": analysis_data.mfu_metrics.matmul_mfu if analysis_data.mfu_metrics else 0,
                    "attention_mfu": analysis_data.mfu_metrics.attention_mfu if analysis_data.mfu_metrics else 0,
                    "operator_count": analysis_data.total_operators,
                    "low_efficiency_count": len(analysis_data.low_efficiency_operators),
                    "fusion_opportunity_count": len(analysis_data.fusion_opportunities),
                    "fusion_opportunities": [
                        {
                            "name": opp.name,
                            "type": opp.opportunity_type,
                            "speedup": opp.estimated_speedup,
                            "end_to_end_speedup": opp.end_to_end_speedup,
                            "time_proportion": opp.time_proportion,
                            "complexity": opp.complexity,
                        }
                        for opp in analysis_data.fusion_opportunities[:5]
                    ],
                },
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Operator analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="算子分析失败",
                error=str(e),
            )

    def _prepare_analysis_data(self, data: Dict[str, Any]) -> OperatorAnalysisData:
        """准备分析数据"""
        analysis_data = OperatorAnalysisData()

        # 芯片信息
        if "chip_info" in data:
            chip_info = data["chip_info"]
            if isinstance(chip_info, ChipInfo):
                analysis_data.chip_name = chip_info.chip_name
                analysis_data.peak_flops_tflops = chip_info.get_peak_flops() / 1e12
        elif "profiling_path" in data:
            chip_info = ChipInfo.from_profiling_path(data["profiling_path"])
            analysis_data.chip_name = chip_info.chip_name
            analysis_data.peak_flops_tflops = chip_info.get_peak_flops() / 1e12

        # Top 算子
        if "top_operators" in data:
            analysis_data.top_operators = data["top_operators"]
            analysis_data.total_operators = len(data["top_operators"])

        return analysis_data

    def _calculate_mfu(
        self,
        operators_df: pd.DataFrame,
        chip_info: Optional[ChipInfo] = None,
    ) -> Optional[MFUMetrics]:
        """计算 MFU"""
        if operators_df.empty:
            return None

        if chip_info is None:
            chip_info = ChipInfo.default_ascend_910b()

        calculator = MFUCalculator(chip_info)
        return calculator.analyze_operators(operators_df)

    def _extract_recommendations(self, response: str) -> List[str]:
        """从 LLM 响应中提取优化建议"""
        recommendations = []

        lines = response.split("\n")
        in_recommendation = False

        for line in lines:
            line_lower = line.lower()
            if "建议" in line or "优化" in line or "suggestion" in line_lower:
                in_recommendation = True
            if in_recommendation and line.strip().startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                clean_line = line.strip().lstrip("-*•0123456789. )")
                if clean_line and len(clean_line) > 5:
                    recommendations.append(clean_line)

        return recommendations[:10]
