"""
Comparison Advisor Agent

专门用于分析两个 Profiling 差异数据，
利用 LLM 进行深度根因分析和优化建议生成。
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.agents.base_agent import BaseAgent, AnalysisResult
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


COMPARISON_SYSTEM_PROMPT = """你是一位资深的昇腾 NPU 大模型训练性能对比分析专家。

你的任务是分析两次 Profiling 数据的差异，找出性能变化的根本原因，并给出针对性的优化建议。

分析框架：
1. **变化识别**：明确指出哪些指标发生了显著变化（>5%）
2. **根因分析**：基于算子变化、通信模式变化、时间分布变化等数据，推断导致性能变化的根本原因
3. **影响评估**：量化各因素对整体性能的影响程度
4. **优化建议**：针对性能劣化的维度，给出具体、可操作的优化方案

常见性能变化原因：
- 软件版本升级导致算子实现变化（CANN 版本、PTA 版本）
- 并行策略调整（TP/PP/DP 配比变化、通信组重构）
- 模型结构微调（新增/替换算子、改变计算图）
- 硬件配置变化（不同芯片型号、不同卡数）
- 通信拓扑变化（HCCL 配置、Ring/Mesh 策略）
- 内存策略变化（重计算、ZeRO level、梯度累积）

输出要求：
- 使用 Markdown 格式
- 关键数据用表格展示
- 按影响程度排序分析结论
- 优化建议需具体到参数/配置/代码层面
- 给出预期收益的量化估计
"""


class ComparisonAdvisorAgent(BaseAgent):
    """
    Profiling 对比分析 Agent

    接收 ProfilingDiff 结果，利用 LLM 进行深度根因分析。
    在 LLM 不可用（mock backend）时，使用规则引擎进行分析。
    """

    PROMPT_TEMPLATE = """## 对比分析任务

### 基本信息
- **基准版本 (A)**: {label_a}
  - 路径: `{path_a}`
  - Rank 数: {rank_a}，Step 数: {step_a}
- **当前版本 (B)**: {label_b}
  - 路径: `{path_b}`
  - Rank 数: {rank_b}，Step 数: {step_b}

### 相似度评估
{similarity_text}

### 差异分析数据
{diff_text}

### 分析要求
请基于以上差异数据，完成以下分析：

1. **性能变化总结**（一段话概括核心变化）
2. **根因分析**（列出导致性能变化的可能原因，按影响程度排序）
3. **详细对比分析**
   - 计算效率变化分析
   - 通信效率变化分析
   - 算子级别变化分析（关注 Top 劣化/改善算子）
   - 空闲时间变化分析
4. **优化建议**（按优先级高/中/低分类）
5. **预期收益**（如果执行建议后的预期性能提升）
"""

    def __init__(self, llm: LLMInterface, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="ComparisonAdvisorAgent",
            llm=llm,
            system_prompt=COMPARISON_SYSTEM_PROMPT,
            config=config,
        )

    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        执行对比分析

        Args:
            data: 包含以下字段:
                - diff: ProfilingDiff 对象或其 dict 表示
                - diff_text: ProfilingDiff.to_prompt_text() 的文本
                - similarity_text: SimilarityResult.to_markdown() 的文本
                - label_a / label_b: 版本标签
                - path_a / path_b: Profiling 路径
                - summary_a / summary_b: ProfilingSummary 的 dict
                - rank_a / rank_b: Rank 数
                - step_a / step_b: Step 数

        Returns:
            AnalysisResult
        """
        try:
            # 构建 Prompt
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                label_a=data.get("label_a", "基准版本"),
                label_b=data.get("label_b", "当前版本"),
                path_a=data.get("path_a", "N/A"),
                path_b=data.get("path_b", "N/A"),
                rank_a=data.get("rank_a", "N/A"),
                rank_b=data.get("rank_b", "N/A"),
                step_a=data.get("step_a", "N/A"),
                step_b=data.get("step_b", "N/A"),
                similarity_text=data.get("similarity_text", "N/A"),
                diff_text=data.get("diff_text", "N/A"),
            )

            # 调用 LLM
            response = await self.call_llm(prompt)

            # 提取建议
            recommendations = self._extract_recommendations(response)

            # 规则分析（作为 LLM 分析的补充）
            rule_insights = self._rule_based_analysis(data)

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary="对比分析完成",
                details={
                    "llm_analysis": response,
                    "rule_insights": rule_insights,
                },
                recommendations=recommendations,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Comparison analysis failed: {e}", exc_info=True)

            # 降级到纯规则分析
            rule_insights = self._rule_based_analysis(data)
            return AnalysisResult(
                agent_name=self.name,
                success=True,  # 规则分析依然可用
                summary="对比分析完成（规则引擎）",
                details={
                    "rule_insights": rule_insights,
                },
                recommendations=rule_insights,
                raw_response="\n".join(f"- {r}" for r in rule_insights),
                error=str(e) if not rule_insights else None,
            )

    def _rule_based_analysis(self, data: Dict[str, Any]) -> List[str]:
        """
        基于规则的分析（LLM 不可用时的降级方案）

        Returns:
            分析结论和建议列表
        """
        insights = []
        diff_data = data.get("diff", {})
        if hasattr(diff_data, "to_dict"):
            diff_data = diff_data.to_dict()

        if not diff_data:
            return insights

        # 1. Summary 级别规则
        summary_diff = diff_data.get("summary_diff", {})
        changes = summary_diff.get("changes", [])

        for change in changes:
            name = change.get("name", "")
            pct = change.get("change_pct", 0)
            is_improvement = change.get("is_improvement", False)
            significance = change.get("significance", "low")

            if significance in ("medium", "high"):
                label = change.get("label", name)
                if name == "step_time" and not is_improvement:
                    insights.append(
                        f"Step 时间增加 {abs(pct):.1f}%，建议检查: "
                        f"(1) 软件版本更新导致的算子性能退化; "
                        f"(2) 新增的空闲等待时间; "
                        f"(3) 通信模式变化"
                    )
                elif name == "idle_ratio" and not is_improvement:
                    insights.append(
                        f"空闲时间占比增加 {abs(pct):.1f}%，可能原因: "
                        f"Host-Device 同步增加、数据加载瓶颈、调度效率下降"
                    )
                elif name == "comm_ratio" and not is_improvement:
                    insights.append(
                        f"通信占比增加 {abs(pct):.1f}%，建议检查: "
                        f"并行策略配置、HCCL 通信算法、通信掩盖策略"
                    )
                elif name == "overlap_ratio" and not is_improvement:
                    insights.append(
                        f"通信掩盖率下降 {abs(pct):.1f}%，建议: "
                        f"(1) 启用 async grad reduce; "
                        f"(2) 调整 micro_batch 数量增加计算窗口; "
                        f"(3) 检查通信调度策略"
                    )

        # 2. Operator 级别规则
        operator_diff = diff_data.get("operator_diff", {})
        top_regressions = operator_diff.get("top_regressions", [])

        if top_regressions:
            top_names = [r["name"] for r in top_regressions[:3]]
            insights.append(
                f"Top 劣化算子: {', '.join(top_names)}。"
                f"建议通过 msprof op --aic-metrics 分析具体算子的硬件利用率"
            )

        new_ops = operator_diff.get("new_operators", [])
        if new_ops:
            insights.append(
                f"新增 {len(new_ops)} 个算子，可能由软件版本更新或模型结构变化引入。"
                f"建议检查新增算子是否有更高效的实现"
            )

        # 3. Timeline 级别规则
        timeline_diff = diff_data.get("timeline_diff", {})
        if timeline_diff:
            cv_a = timeline_diff.get("step_time_cv_a", 0)
            cv_b = timeline_diff.get("step_time_cv_b", 0)
            if cv_b > cv_a * 1.5 and cv_b > 0.05:
                insights.append(
                    f"Step 时间波动增大 (CV: {cv_a:.4f} → {cv_b:.4f})，"
                    f"可能存在新的 Jitter 源（如 GC、数据加载抖动、通信拥塞）"
                )

        if not insights:
            verdict = diff_data.get("overall_verdict", "unchanged")
            if verdict == "unchanged":
                insights.append("两次 Profiling 性能基本一致，无显著差异")
            elif verdict == "improved":
                insights.append("性能有所提升，优化措施有效")

        return insights
