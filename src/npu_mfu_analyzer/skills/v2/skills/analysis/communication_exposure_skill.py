"""
通信暴露诊断技能

将通信/overlap/bubble 结果映射为更稳定的暴露类型和优先动作。
"""

import logging
from typing import Any, Dict, List

from npu_mfu_analyzer.agents.base_agent import AnalysisResult
from npu_mfu_analyzer.skills.v2.base import (
    BaseSkill,
    SkillType,
    SkillCategory,
    SkillMetadata,
    SkillPriority,
    SkillContext,
    SkillResult,
)

logger = logging.getLogger(__name__)


class CommunicationExposureSkill(BaseSkill):
    """通信暴露诊断技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="diagnose_communication_exposure",
            display_name="通信暴露诊断",
            description="识别当前 profiling 更像哪类通信在暴露，以及 overlap 是否失效",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.HIGH,
            tags=["communication", "overlap", "bubble", "tp", "pp", "hsdp"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            hints = summary.get("training_hints", {}) or {}
            topology = context.user_inputs.get("topology_summary") or summary.get("topology_summary") or {}
            agent_results = context.user_inputs.get("agent_results", {})
            timeline_result = self._get_agent_result(agent_results, "timeline")
            comm_result = self._get_agent_result(agent_results, "communication")

            avg_step = self._to_float(summary.get("avg_step_time"))
            avg_comm = self._to_float(summary.get("avg_comm_time"))
            comm_ratio = self._to_float(self._detail(comm_result, "comm_ratio"))
            if comm_ratio <= 0 and avg_step > 0:
                comm_ratio = avg_comm / avg_step

            overlap_ratio = self._first_non_none(
                self._detail(timeline_result, "overlap_ratio"),
                self._nested(summary, "overlap_metrics", "overlap_ratio"),
                0.0,
            )
            overlap_ratio = self._to_float(overlap_ratio)
            bubble_ratio = self._to_float(self._first_non_none(
                self._detail(timeline_result, "bubble_ratio"),
                summary.get("bubble_ratio"),
                0.0,
            ))

            tp_comm_ms = self._to_float(self._detail(comm_result, "tp_comm_time_ms"))
            pp_comm_ms = self._to_float(self._detail(comm_result, "pp_comm_time_ms"))
            dp_comm_ms = self._to_float(self._detail(comm_result, "dp_comm_time_ms"))
            slow_ranks = self._detail(comm_result, "slow_ranks") or []

            evidence: List[str] = []
            if comm_ratio > 0:
                evidence.append(f"通信占比 {comm_ratio * 100:.1f}%")
            if overlap_ratio > 0:
                evidence.append(f"通信掩盖率 {overlap_ratio:.1f}%")
            if bubble_ratio > 0:
                evidence.append(f"Pipeline bubble 占比 {bubble_ratio:.1f}%")
            if tp_comm_ms > 0 or pp_comm_ms > 0 or dp_comm_ms > 0:
                evidence.append(
                    f"通信拆分 TP/PP/DP={tp_comm_ms:.2f}/{pp_comm_ms:.2f}/{dp_comm_ms:.2f} ms"
                )
            if slow_ranks:
                evidence.append(f"检测到慢卡: {slow_ranks}")
            if topology.get("num_machines", 1) > 1:
                evidence.append(
                    f"拓扑: {topology.get('num_machines')} 机, 跨节点通信比例 {float(topology.get('inter_node_ratio', 0))*100:.1f}%"
                )

            dominant_domain = "generic"
            dominant_value = max(tp_comm_ms, pp_comm_ms, dp_comm_ms, 0.0)
            if dominant_value > 0:
                if dominant_value == tp_comm_ms:
                    dominant_domain = "tp"
                elif dominant_value == pp_comm_ms:
                    dominant_domain = "pp"
                else:
                    dominant_domain = "dp"

            diagnosis_code = "UNKNOWN_COMM_PATTERN"
            layer = "unknown"
            confidence = 0.35
            recommendations: List[str] = []
            actions: List[Dict[str, Any]] = []

            if bubble_ratio >= 20:
                diagnosis_code = "PIPELINE_BUBBLE_BOUND"
                layer = "parallelism"
                confidence = 0.85
                recommendations.extend([
                    "当前更像 PP bubble 或 stage 不均衡问题。",
                    "优先增加 microbatch 数、检查 stage 负载，再考虑 VPP/Interleaved。",
                ])
                actions.extend([
                    {
                        "title": "优先检查 PP microbatch 与 stage 均衡",
                        "action_code": "tune_pipeline_schedule",
                        "layer": "parallelism",
                        "priority": "high",
                        "description": "先检查 microbatch 数是否足够、stage 是否均衡，再决定是否上 VPP/Interleaved。",
                        "expected_effect": "降低 bubble，占满流水线稳态区间。",
                        "rationale": "bubble 占比已经偏高。",
                        "anti_actions": ["当前不建议先做 optimizer-state-offload", "当前不建议先改 activation swap"],
                    }
                ])
            elif comm_ratio >= 0.25 and hints.get("moe_likely") and slow_ranks:
                diagnosis_code = "EP_COMM_EXPOSED"
                layer = "parallelism"
                confidence = 0.78
                recommendations.extend([
                    "当前更像 MoE dispatch/combine 或 expert 负载不均导致的通信暴露。",
                    "优先检查 router 热点、expert 负载均衡和 all-to-all overlap。",
                ])
                actions.extend([{
                    "title": "优先检查 EP all-to-all 与 expert 负载均衡",
                    "action_code": "tune_ep_communication",
                    "layer": "parallelism",
                    "priority": "high",
                    "description": "先确认 router/load-balance，再处理 all-to-all overlap。",
                    "expected_effect": "减少 expert 侧通信与慢 rank 拖尾。",
                    "rationale": "MoE 问题常同时包含通信和负载不均。",
                    "anti_actions": ["当前不建议先扩大 EP/TP 度数"],
                }])
            elif comm_ratio >= 0.25 and hints.get("long_context_likely") and dominant_domain in {"generic", "tp"}:
                diagnosis_code = "CP_COMM_EXPOSED"
                layer = "parallelism"
                confidence = 0.72
                recommendations.extend([
                    "当前更像长上下文 attention 相关通信暴露。",
                    "优先检查 CP 布局、A2A/gather 和 overlap。",
                ])
                actions.extend([{
                    "title": "优先检查 CP / Attention 通信路径",
                    "action_code": "tune_cp_communication",
                    "layer": "parallelism",
                    "priority": "high",
                    "description": "先确认长上下文 attention 的通信布局，再处理 overlap。",
                    "expected_effect": "降低 attention 关键路径上的通信暴露。",
                    "rationale": "长序列通信首先来自 attention/CP。",
                    "anti_actions": ["当前不建议先改 optimizer-state-offload"],
                }])
            elif comm_ratio >= 0.20 and overlap_ratio < 50:
                diagnosis_code = "OVERLAP_INEFFECTIVE"
                layer = "system_optimization"
                confidence = 0.85
                recommendations.extend([
                    "通信已明显暴露在关键路径上，且 overlap 效果偏弱。",
                    "优先检查 wait 时机、prefetch 时机和是否存在可重叠工作。",
                ])
                actions.extend([
                    {
                        "title": "优先检查 Overlap / Wait 时机",
                        "action_code": "improve_overlap",
                        "layer": "system_optimization",
                        "priority": "high",
                        "description": "把 wait 推迟到真正消费结果前，并确认异步发起后确实有独立计算可重叠。",
                        "expected_effect": "减少纯等待通信时间。",
                        "rationale": "当前 overlap ratio 偏低且通信占比不低。",
                        "anti_actions": ["当前不建议先扩大并行度", "当前不建议先做大规模策略切换"],
                    },
                    {
                        "title": "如通信过碎，再评估 Comm Fusion",
                        "action_code": "enable_comm_fusion",
                        "layer": "system_optimization",
                        "priority": "medium",
                        "description": "如果时间线上是很多小通信反复暴露，应优先合并通信粒度。",
                        "expected_effect": "降低小通信 launch/latency 开销。",
                        "rationale": "overlap 之外，碎通信也是常见二级矛盾。",
                        "anti_actions": ["不要直接把问题归因到模型结构"],
                    },
                ])
            elif comm_ratio >= 0.25 and dominant_domain == "tp":
                diagnosis_code = "TP_COMM_EXPOSED"
                layer = "parallelism"
                confidence = 0.75
                recommendations.extend([
                    "当前更像 TP 层内通信暴露。",
                    "优先检查 TP 组拓扑局部性、SequenceParallel 和通信融合。",
                ])
                actions.extend([
                    {
                        "title": "优先优化 TP 通信局部性与粒度",
                        "action_code": "tune_tp_communication",
                        "layer": "parallelism",
                        "priority": "high",
                        "description": "先确认 TP group 是否落在低时延域内，再看 SequenceParallel、comm fusion 是否到位。",
                        "expected_effect": "降低每层 TP collective 暴露。",
                        "rationale": "TP 通信通常是高频关键路径。",
                        "anti_actions": ["当前不建议先上更多 PP"],
                    }
                ])
            elif comm_ratio >= 0.25 and dominant_domain == "dp":
                diagnosis_code = "HSDP_DP_COMM_EXPOSED"
                layer = "system_optimization"
                confidence = 0.75
                recommendations.extend([
                    "当前更像 HSDP/DP 侧梯度通信暴露。",
                    "优先检查 grad fusion、overlap 和必要时的梯度累积。",
                ])
                actions.extend([
                    {
                        "title": "优先优化 Grad Fusion 与 Reduce 通信掩盖",
                        "action_code": "tune_grad_communication",
                        "layer": "system_optimization",
                        "priority": "high",
                        "description": "先合并碎 grad 通信并后移 wait，再决定是否调整 data/shard 配置。",
                        "expected_effect": "降低 backward 尾部通信暴露。",
                        "rationale": "DP/HSDP 通信通常集中暴露在 backward/grad-state 段。",
                        "anti_actions": ["当前不建议先做 activation swap"],
                    }
                ])
            elif comm_ratio >= 0.25:
                diagnosis_code = "COMMUNICATION_EXPOSED"
                layer = "system_optimization"
                confidence = 0.65
                recommendations.append("通信已经是主瓶颈之一，优先做 overlap / fusion / topology 侧优化。")

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "diagnosis_code": diagnosis_code,
                    "layer": layer,
                    "comm_ratio": comm_ratio,
                    "overlap_ratio": overlap_ratio,
                    "bubble_ratio": bubble_ratio,
                    "dominant_domain": dominant_domain,
                    "observation_facts": evidence,
                    "candidate_actions": actions,
                },
                summary=f"通信归因: {diagnosis_code}",
                details=evidence,
                recommendations=recommendations,
                actions=actions,
                root_cause=f"当前更像 {diagnosis_code}",
                evidence=evidence,
                confidence=confidence,
                priority="P0" if diagnosis_code not in {"UNKNOWN_COMM_PATTERN"} else "P2",
                severity="high" if comm_ratio >= 0.30 else "medium",
            )
        except Exception as e:
            logger.exception("通信暴露诊断失败: %s", e)
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )

    def _get_agent_result(self, agent_results: Dict[str, Any], key: str) -> Any:
        return agent_results.get(key)

    def _detail(self, result: Any, key: str) -> Any:
        if isinstance(result, AnalysisResult):
            return result.details.get(key)
        if isinstance(result, dict):
            return result.get("details", {}).get(key, result.get(key))
        return None

    def _to_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value if isinstance(value, dict) else {}

    def _nested(self, data: Dict[str, Any], key: str, field: str) -> Any:
        value = data.get(key, {})
        if isinstance(value, dict):
            return value.get(field)
        return None

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _first_non_none(self, *values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None
