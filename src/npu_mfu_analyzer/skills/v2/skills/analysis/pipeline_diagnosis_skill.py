"""PP/Pipeline 专项诊断技能。"""

import logging
from typing import Any, Dict, List

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class PipelineDiagnosisSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="diagnose_pipeline_behavior",
            display_name="Pipeline 专项诊断",
            description="对 bubble / microbatch / stage imbalance 做专项诊断",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.COMMUNICATION,
            priority=SkillPriority.HIGH,
            tags=["pipeline", "pp", "bubble"],
            dependencies=["diagnose_communication_exposure"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            comm_result = context.get_previous_result("diagnose_communication_exposure")
            comm_code = comm_result.data.get("diagnosis_code", "") if comm_result and comm_result.success else ""
            bubble_ratio = self._to_float(summary.get("bubble_ratio"))
            step_count = int(summary.get("step_count", 0) or 0)
            rank_count = int(summary.get("rank_count", 1) or 1)

            actions: List[Dict[str, Any]] = []
            evidence: List[str] = []
            diagnosis_code = "PIPELINE_NOT_PRIMARY"
            confidence = 0.40

            if bubble_ratio >= 15 or comm_code == "PIPELINE_BUBBLE_BOUND":
                diagnosis_code = "PIPELINE_BUBBLE_BOUND"
                confidence = 0.85
                evidence.append(f"bubble 占比 {bubble_ratio:.1f}%")
                if step_count > 0 and rank_count > 1:
                    evidence.append(f"step 数 {step_count}，rank 数 {rank_count}")
                actions.extend([
                    {
                        "title": "优先检查 microbatch 数与 stage 负载均衡",
                        "action_code": "tune_pipeline_microbatch",
                        "layer": "parallelism",
                        "priority": "high",
                        "description": "先确认 microbatch 是否足够灌满流水线，再检查 stage 是否失衡。",
                        "expected_effect": "降低 PP bubble，提升稳态利用率。",
                        "rationale": "bubble 高通常先是 schedule/microbatch/stage 组合问题。",
                        "anti_actions": ["当前不建议先做 optimizer-state-offload"],
                    },
                    {
                        "title": "必要时再评估 VPP / Interleaved",
                        "action_code": "consider_vpp",
                        "layer": "parallelism",
                        "priority": "medium",
                        "description": "在 stage 负载已基本均衡、microbatch 已合理时，再评估更细粒度 pipeline 调度。",
                        "expected_effect": "进一步压缩 bubble。",
                        "rationale": "VPP 是二级动作，不应先于基础 schedule 诊断。",
                        "anti_actions": ["不要先用 activation swap 替代 pipeline 调整"],
                    },
                ])

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "diagnosis_code": diagnosis_code,
                    "observation_facts": evidence,
                    "candidate_actions": actions,
                },
                summary=f"Pipeline 诊断: {diagnosis_code}",
                details=evidence,
                recommendations=[a["title"] for a in actions],
                actions=actions,
                evidence=evidence,
                confidence=confidence,
                priority="P1" if diagnosis_code != "PIPELINE_NOT_PRIMARY" else "P2",
            )
        except Exception as e:
            logger.exception("Pipeline 诊断失败: %s", e)
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
