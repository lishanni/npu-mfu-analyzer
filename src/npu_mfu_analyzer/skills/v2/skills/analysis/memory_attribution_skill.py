"""
内存归因技能

将显存相关观测映射为 activation-bound / state-bound 等更稳定的诊断对象。
"""

import logging
from typing import Any, Dict, List, Tuple

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


class MemoryAttributionSkill(BaseSkill):
    """内存主矛盾归因技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="attribute_memory_bottleneck",
            display_name="内存归因",
            description="识别当前 profiling 更像 activation-bound 还是 state-bound",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.MEMORY,
            priority=SkillPriority.HIGH,
            tags=["memory", "activation", "optimizer-state", "diagnosis"],
            dependencies=[],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            agent_results = context.user_inputs.get("agent_results", {})
            memory_result = self._get_agent_result(agent_results, "memory")

            peak_memory_gb = self._first_non_none(
                self._detail(memory_result, "peak_memory_gb"),
                summary.get("peak_memory_gb"),
                self._mb_to_gb(summary.get("peak_memory_mb")),
                0.0,
            )
            memory_utilization = float(self._first_non_none(
                self._detail(memory_result, "memory_utilization"),
                summary.get("memory_utilization"),
                0.0,
            ) or 0.0)
            oom_risk = str(self._first_non_none(
                self._detail(memory_result, "oom_risk"),
                summary.get("oom_risk"),
                "unknown",
            ))

            activation_mb = self._to_float(self._first_non_none(
                summary.get("activation_memory_mb"),
                context.user_inputs.get("activation_memory_mb"),
            ))
            gradient_mb = self._to_float(self._first_non_none(
                summary.get("gradient_memory_mb"),
                context.user_inputs.get("gradient_memory_mb"),
            ))
            optimizer_mb = self._to_float(self._first_non_none(
                summary.get("optimizer_memory_mb"),
                context.user_inputs.get("optimizer_memory_mb"),
            ))
            model_mb = self._to_float(self._first_non_none(
                summary.get("model_memory_mb"),
                context.user_inputs.get("model_memory_mb"),
            ))
            peak_mb = self._to_float(self._first_non_none(
                summary.get("peak_memory_mb"),
                peak_memory_gb * 1024,
            ))

            avg_step = self._to_float(summary.get("avg_step_time"))
            avg_compute = self._to_float(summary.get("avg_compute_time"))
            avg_comm = self._to_float(summary.get("avg_comm_time"))
            compute_ratio = (avg_compute / avg_step) if avg_step > 0 else 0.0
            comm_ratio = (avg_comm / avg_step) if avg_step > 0 else 0.0

            activation_ratio = ((activation_mb + gradient_mb) / peak_mb) if peak_mb > 0 else 0.0
            state_ratio = ((optimizer_mb + model_mb) / peak_mb) if peak_mb > 0 else 0.0
            hints = self._first_non_none(
                self._detail(memory_result, "training_hints"),
                summary.get("training_hints"),
                {},
            ) or {}
            activation_hint_score = self._to_float(hints.get("activation_pressure_score"))
            state_hint_score = self._to_float(hints.get("state_pressure_score"))

            diagnosis_code = "UNKNOWN_MEMORY_PATTERN"
            confidence = 0.35
            evidence: List[str] = []
            recommendations: List[str] = []
            actions: List[Dict[str, Any]] = []

            if peak_memory_gb > 0:
                evidence.append(f"峰值显存 {peak_memory_gb:.2f} GB")
            if hints.get("long_context_likely"):
                evidence.append("training hints: 更像长序列 dense")
            if hints.get("moe_likely"):
                evidence.append("training hints: 更像 MoE")
            if memory_utilization > 0:
                evidence.append(f"内存利用率 {memory_utilization:.1f}%")
            if oom_risk != "unknown":
                evidence.append(f"OOM 风险 {oom_risk}")
            if activation_mb > 0 or gradient_mb > 0:
                evidence.append(
                    f"activation+gradient {activation_mb + gradient_mb:.0f} MB，占峰值 {activation_ratio*100:.1f}%"
                )
            if optimizer_mb > 0 or model_mb > 0:
                evidence.append(
                    f"model+optimizer {optimizer_mb + model_mb:.0f} MB，占峰值 {state_ratio*100:.1f}%"
                )

            if activation_ratio >= max(state_ratio + 0.10, 0.35):
                diagnosis_code = "ACTIVATION_MEMORY_BOUND"
                confidence = 0.9 if activation_mb > 0 else 0.7
            elif state_ratio >= max(activation_ratio + 0.10, 0.45):
                diagnosis_code = "STATE_MEMORY_BOUND"
                confidence = 0.9 if optimizer_mb > 0 or model_mb > 0 else 0.7
            elif activation_hint_score >= max(state_hint_score + 0.15, 0.45):
                diagnosis_code = "ACTIVATION_MEMORY_BOUND"
                confidence = 0.65
                evidence.append(f"operator/training hints 指向 activation 压力 ({activation_hint_score:.2f})")
            elif state_hint_score >= max(activation_hint_score + 0.15, 0.40):
                diagnosis_code = "STATE_MEMORY_BOUND"
                confidence = 0.60
                evidence.append(f"operator/training hints 指向 state 压力 ({state_hint_score:.2f})")
            elif oom_risk == "high" and compute_ratio >= 0.55 and comm_ratio <= 0.25:
                diagnosis_code = "ACTIVATION_MEMORY_BOUND"
                confidence = 0.6
                evidence.append("高 OOM 风险 + 计算占比高 + 通信占比低，更像 activation 峰值主导")
            elif oom_risk == "high" and memory_utilization >= 85:
                diagnosis_code = "STATE_MEMORY_BOUND"
                confidence = 0.55
                evidence.append("高 OOM 风险 + 高常驻显存利用率，但缺少更细内存拆分，暂按 state-bound 低置信推断")

            if diagnosis_code == "ACTIVATION_MEMORY_BOUND":
                recommendations.extend([
                    "当前更像 activation 峰值主导，优先检查 recompute 粒度与 activation 生命周期。",
                    "如果重计算代价过高，再考虑 activation swap/offload。",
                ])
                actions.extend([
                    {
                        "title": "优先启用或加深 Recompute",
                        "action_code": "enable_recompute",
                        "layer": "memory_strategy",
                        "priority": "high",
                        "description": "先降低 forward/backward 峰值 activation，而不是先碰 optimizer state。",
                        "expected_effect": "降低 activation 峰值显存，缓解 forward/backward OOM 风险。",
                        "rationale": "当前证据更像 activation-bound。",
                        "anti_actions": ["当前不建议先改 optimizer-state-offload", "当前不建议先调大 TP/PP 规模"],
                    },
                    {
                        "title": "如重计算代价过高，再评估 Activation Swap",
                        "action_code": "enable_activation_swap",
                        "layer": "memory_strategy",
                        "priority": "medium",
                        "description": "在算力已经很紧、但 Host 带宽尚可时，再考虑 activation swap。",
                        "expected_effect": "进一步压低 activation 峰值。",
                        "rationale": "swap 是 activation 问题的第二选择，不应先于 recompute。",
                        "anti_actions": ["不要先把问题归到 optimizer state"],
                    },
                ])
            elif diagnosis_code == "STATE_MEMORY_BOUND":
                recommendations.extend([
                    "当前更像参数/优化器状态常驻带来的显存压力。",
                    "优先考虑 optimizer state offload，其次再看 parameter offload。",
                ])
                actions.extend([
                    {
                        "title": "优先评估 Optimizer State Offload",
                        "action_code": "enable_optimizer_state_offload",
                        "layer": "memory_strategy",
                        "priority": "high",
                        "description": "先处理低频大对象 optimizer state，而不是先对关键路径参数做 offload。",
                        "expected_effect": "降低 step 阶段状态显存峰值。",
                        "rationale": "当前证据更像 state-bound。",
                        "anti_actions": ["当前不建议先做 activation swap", "当前不建议优先调整 CP/EP"],
                    },
                    {
                        "title": "如基线仍高，再评估 Parameter Offload / HSDP 收紧",
                        "action_code": "enable_parameter_offload",
                        "layer": "memory_strategy",
                        "priority": "medium",
                        "description": "在 optimizer state 处理后，如果基线仍然过高，再评估参数驻留策略。",
                        "expected_effect": "降低设备常驻参数基线。",
                        "rationale": "参数 offload 更贴近关键路径，优先级应低于 optimizer state。",
                        "anti_actions": ["不要先加重 recompute 作为主动作"],
                    },
                ])
            else:
                recommendations.append("当前内存证据不足，建议补充更细的峰值阶段数据或 activation/state 拆分后再下结论。")

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "diagnosis_code": diagnosis_code,
                    "layer": "memory_strategy" if diagnosis_code != "UNKNOWN_MEMORY_PATTERN" else "unknown",
                    "peak_memory_gb": peak_memory_gb,
                    "memory_utilization": memory_utilization,
                    "oom_risk": oom_risk,
                    "activation_ratio": activation_ratio,
                    "state_ratio": state_ratio,
                    "observation_facts": evidence,
                    "candidate_actions": actions,
                    "activation_hint_score": activation_hint_score,
                    "state_hint_score": state_hint_score,
                },
                summary=f"内存归因: {diagnosis_code}",
                details=evidence,
                recommendations=recommendations,
                actions=actions,
                root_cause="当前更像 activation-bound" if diagnosis_code == "ACTIVATION_MEMORY_BOUND" else (
                    "当前更像 state-bound" if diagnosis_code == "STATE_MEMORY_BOUND" else "当前内存模式暂不明确"
                ),
                evidence=evidence,
                confidence=confidence,
                priority="P0" if diagnosis_code in {"ACTIVATION_MEMORY_BOUND", "STATE_MEMORY_BOUND"} else "P2",
                severity="high" if oom_risk == "high" else "medium",
            )
        except Exception as e:
            logger.exception("内存归因失败: %s", e)
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

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _mb_to_gb(self, value: Any) -> float:
        return self._to_float(value) / 1024 if value is not None else 0.0

    def _first_non_none(self, *values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None
