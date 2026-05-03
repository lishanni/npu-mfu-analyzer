"""训练场景专项诊断技能。"""

import logging
from typing import Any, Dict, List

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class ScenarioDiagnosisSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="diagnose_training_scenario",
            display_name="训练场景诊断",
            description="识别 long-context dense / MoE / pipeline dense 等场景并给出专项动作",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            tags=["scenario", "long-context", "moe", "cp", "ep"],
            dependencies=["identify_main_contradiction", "diagnose_communication_exposure", "diagnose_pipeline_behavior"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            hints = summary.get("training_hints", {}) or {}
            main_result = context.get_previous_result("identify_main_contradiction")
            comm_result = context.get_previous_result("diagnose_communication_exposure")
            pipeline_result = context.get_previous_result("diagnose_pipeline_behavior")

            main_code = main_result.data.get("main_contradiction", {}).get("code", "") if main_result and main_result.success else ""
            comm_code = comm_result.data.get("diagnosis_code", "") if comm_result and comm_result.success else ""
            pipeline_code = pipeline_result.data.get("diagnosis_code", "") if pipeline_result and pipeline_result.success else ""

            scenario = "GENERAL_DENSE"
            confidence = 0.45
            evidence: List[str] = []
            actions: List[Dict[str, Any]] = []

            if hints.get("moe_likely"):
                scenario = "MOE_TRAINING"
                confidence = 0.85
                evidence.append("Top 算子中检测到 expert/router/MoE 模式")
                if main_code == "STATE_MEMORY_BOUND":
                    evidence.append("MoE + state-bound，常见于 expert 参数/optimizer state 膨胀")
                if comm_code in {"EP_COMM_EXPOSED", "OVERLAP_INEFFECTIVE"}:
                    evidence.append("MoE dispatch/combine 可能已暴露在关键路径")
                actions.extend([
                    {
                        "title": "优先检查 EP 负载均衡与 Router 热点",
                        "action_code": "tune_ep_load_balance",
                        "layer": "parallelism",
                        "priority": "high",
                        "description": "先看 expert 分布、热点 expert 和 dispatch/combine all-to-all，再决定是否扩大 EP。",
                        "expected_effect": "减少最慢 expert rank 拖尾和 all-to-all 暴露。",
                        "rationale": "MoE 首先要排除 load balance 问题。",
                        "anti_actions": ["当前不建议先放大 TP/PP"],
                    }
                ])
                if main_code == "STATE_MEMORY_BOUND":
                    actions.append({
                        "title": "同步评估 Optimizer State Offload",
                        "action_code": "enable_optimizer_state_offload",
                        "layer": "memory_strategy",
                        "priority": "high",
                        "description": "MoE 常先受 expert optimizer state 影响，应尽早评估状态卸载。",
                        "expected_effect": "降低 step 阶段状态峰值。",
                        "rationale": "MoE 参数/状态规模天然更大。",
                        "anti_actions": ["不要只靠 recompute 处理 state 问题"],
                    })
            elif hints.get("long_context_likely"):
                scenario = "LONG_CONTEXT_DENSE"
                confidence = 0.80
                evidence.append("Top 算子呈现明显 attention/softmax/flash-attention 模式")
                if main_code == "ACTIVATION_MEMORY_BOUND":
                    evidence.append("长序列 + activation-bound，通常先看 CP + recompute")
                actions.extend([
                    {
                        "title": "优先评估 CP 以分担长上下文 Attention",
                        "action_code": "enable_cp",
                        "layer": "parallelism",
                        "priority": "high",
                        "description": "先把长上下文 attention 分布式化，再评估更细调优。",
                        "expected_effect": "降低长序列 attention 的单卡压力。",
                        "rationale": "长上下文问题首先是 attention 分布式化。",
                        "anti_actions": ["当前不建议先做 optimizer-state-offload"],
                    },
                    {
                        "title": "同步启用或加强 Recompute",
                        "action_code": "enable_recompute",
                        "layer": "memory_strategy",
                        "priority": "high",
                        "description": "长序列场景下 activation 峰值通常仍高，recompute 是更稳的一线动作。",
                        "expected_effect": "降低 forward/backward 峰值显存。",
                        "rationale": "CP 解决 attention 可算性，recompute 解决 activation 峰值。",
                        "anti_actions": ["不要先把问题归到 optimizer state"],
                    },
                ])
            elif pipeline_code == "PIPELINE_BUBBLE_BOUND":
                scenario = "PIPELINE_DENSE"
                confidence = 0.75
                evidence.append("当前更像 pipeline 调度问题主导")
                actions.extend(pipeline_result.data.get("candidate_actions", []))

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "scenario_code": scenario,
                    "observation_facts": evidence,
                    "candidate_actions": actions,
                },
                summary=f"训练场景: {scenario}",
                details=evidence,
                recommendations=[a["title"] for a in actions],
                actions=actions,
                evidence=evidence,
                confidence=confidence,
                priority="P1" if scenario != "GENERAL_DENSE" else "P2",
            )
        except Exception as e:
            logger.exception("训练场景诊断失败: %s", e)
            return SkillResult(skill_name=self.name, skill_type=self.skill_type, success=False, error=str(e))

    def _to_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value if isinstance(value, dict) else {}
