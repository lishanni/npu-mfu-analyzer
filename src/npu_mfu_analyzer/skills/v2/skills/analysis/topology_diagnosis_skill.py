"""拓扑感知诊断技能。"""

import logging
from typing import Any, Dict, List

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class TopologyDiagnosisSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="diagnose_topology_pressure",
            display_name="拓扑感知诊断",
            description="根据通信拓扑与跨节点比例给出并行放置建议",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.NORMAL,
            tags=["topology", "tp", "pp", "placement"],
            dependencies=["diagnose_communication_exposure"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            topology = context.user_inputs.get("topology_summary") or summary.get("topology_summary") or {}
            comm_result = context.get_previous_result("diagnose_communication_exposure")
            comm_data = comm_result.data if comm_result and comm_result.success else {}
            dominant = comm_data.get("dominant_domain", "generic")

            num_machines = int(topology.get("num_machines", 1) or 1)
            inter_ratio = self._to_float(topology.get("inter_node_ratio"))
            slow_links = int(topology.get("slow_links_count", 0) or 0)
            evidence: List[str] = []
            actions: List[Dict[str, Any]] = []
            diagnosis_code = "TOPOLOGY_NEUTRAL"
            confidence = 0.40

            if num_machines > 1 and dominant == "tp" and inter_ratio >= 0.30:
                diagnosis_code = "TP_CROSS_NODE_RISK"
                confidence = 0.80
                evidence.append(f"跨节点通信比例 {inter_ratio * 100:.1f}%")
                actions.append({
                    "title": "优先把 TP 约束在节点内，把 PP/DP 放到节点间",
                    "action_code": "place_tp_intra_node",
                    "layer": "parallelism",
                    "priority": "high",
                    "description": "TP 对时延极敏感，若已跨节点，优先重排并行映射。",
                    "expected_effect": "降低每层 TP collective 暴露。",
                    "rationale": "TP 一般应优先使用节点内高速互联。",
                    "anti_actions": ["当前不建议先增大 TP 度数"],
                })
            elif slow_links > 0:
                diagnosis_code = "SLOW_LINKS_PRESENT"
                confidence = 0.70
                evidence.append(f"检测到 {slow_links} 条慢链路")
                actions.append({
                    "title": "优先检查慢链路与通信拓扑映射",
                    "action_code": "inspect_slow_links",
                    "layer": "system_optimization",
                    "priority": "medium",
                    "description": "优先排查跨机/跨 ring 慢链路，再决定是否调并行配置。",
                    "expected_effect": "减少异常链路对整体训练步长的拖累。",
                    "rationale": "慢链路会放大所有集合通信暴露。",
                    "anti_actions": ["不要先盲目加更多 overlap 开关"],
                })

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "diagnosis_code": diagnosis_code,
                    "observation_facts": evidence,
                    "candidate_actions": actions,
                },
                summary=f"拓扑诊断: {diagnosis_code}",
                details=evidence,
                recommendations=[a["title"] for a in actions],
                actions=actions,
                evidence=evidence,
                confidence=confidence,
                priority="P1" if diagnosis_code != "TOPOLOGY_NEUTRAL" else "P2",
            )
        except Exception as e:
            logger.exception("拓扑诊断失败: %s", e)
            return SkillResult(skill_name=self.name, skill_type=self.skill_type, success=False, error=str(e))

    def _to_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value if isinstance(value, dict) else {}

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
