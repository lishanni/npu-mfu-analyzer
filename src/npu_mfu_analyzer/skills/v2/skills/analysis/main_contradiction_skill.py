"""
主矛盾识别技能

消化 memory / communication 两条诊断链，输出统一主矛盾和优先动作。
"""

import logging
from typing import Any, Dict, List

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


class MainContradictionSkill(BaseSkill):
    """主矛盾识别技能"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="identify_main_contradiction",
            display_name="主矛盾识别",
            description="统一识别当前 profiling 的主矛盾，并给出优先调优动作",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.CRITICAL,
            tags=["diagnosis", "root-cause", "optimization"],
            dependencies=[
                "attribute_memory_bottleneck",
                "diagnose_communication_exposure",
            ],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            memory_result = context.get_previous_result("attribute_memory_bottleneck")
            comm_result = context.get_previous_result("diagnose_communication_exposure")

            memory_data = memory_result.data if memory_result and memory_result.success else {}
            comm_data = comm_result.data if comm_result and comm_result.success else {}

            memory_code = memory_data.get("diagnosis_code", "UNKNOWN_MEMORY_PATTERN")
            memory_conf = memory_result.confidence if memory_result else 0.0
            comm_code = comm_data.get("diagnosis_code", "UNKNOWN_COMM_PATTERN")
            comm_conf = comm_result.confidence if comm_result else 0.0

            main = {
                "code": "INSUFFICIENT_EVIDENCE",
                "layer": "unknown",
                "reason": "当前结构化证据不足，暂无法稳定判断主矛盾。",
                "confidence": 0.3,
            }
            supporting_facts: List[str] = []
            actions: List[Dict[str, Any]] = []

            if comm_code == "PIPELINE_BUBBLE_BOUND" and comm_conf >= 0.75:
                main = {
                    "code": comm_code,
                    "layer": "parallelism",
                    "reason": "当前更像 PP bubble 或 stage 负载不均导致的流水线效率问题。",
                    "confidence": comm_conf,
                }
                supporting_facts.extend(comm_data.get("observation_facts", []))
                actions = comm_data.get("candidate_actions", [])
            elif memory_code in {"ACTIVATION_MEMORY_BOUND", "STATE_MEMORY_BOUND"} and memory_conf >= 0.6:
                main = {
                    "code": memory_code,
                    "layer": "memory_strategy",
                    "reason": "当前更像显存对象本身是主矛盾，应优先从内存策略层收敛动作。",
                    "confidence": memory_conf,
                }
                supporting_facts.extend(memory_data.get("observation_facts", []))
                actions = memory_data.get("candidate_actions", [])
            elif comm_code not in {"UNKNOWN_COMM_PATTERN"} and comm_conf >= 0.6:
                main = {
                    "code": comm_code,
                    "layer": comm_data.get("layer", "system_optimization"),
                    "reason": "当前更像通信暴露或 overlap/fusion 问题，应优先从系统优化层处理。",
                    "confidence": comm_conf,
                }
                supporting_facts.extend(comm_data.get("observation_facts", []))
                actions = comm_data.get("candidate_actions", [])
            elif memory_code in {"ACTIVATION_MEMORY_BOUND", "STATE_MEMORY_BOUND"}:
                main = {
                    "code": memory_code,
                    "layer": "memory_strategy",
                    "reason": "内存侧已有弱证据，但还需要更细 profiling 支撑。",
                    "confidence": memory_conf,
                }
                supporting_facts.extend(memory_data.get("observation_facts", []))
                actions = memory_data.get("candidate_actions", [])[:1]

            recommendations = [action["title"] for action in actions if action.get("title")]

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "main_contradiction": main,
                    "supporting_facts": supporting_facts,
                    "prioritized_actions": actions,
                },
                summary=f"主矛盾: {main['code']} ({main['layer']})",
                details=supporting_facts,
                recommendations=recommendations,
                actions=actions,
                root_cause=main["reason"],
                evidence=supporting_facts,
                confidence=main["confidence"],
                priority="P0" if main["code"] != "INSUFFICIENT_EVIDENCE" else "P2",
                severity="high" if main["confidence"] >= 0.75 else "medium",
            )
        except Exception as e:
            logger.exception("主矛盾识别失败: %s", e)
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=False,
                error=str(e),
            )
