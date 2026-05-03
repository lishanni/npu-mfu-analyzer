"""Step 阶段归因技能。"""

import logging
from typing import Any, Dict, List

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class StepAttributionSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="attribute_step_bottleneck",
            display_name="Step 阶段归因",
            description="识别当前 profiling 最痛的 step 阶段",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            tags=["step", "phase", "diagnosis"],
            dependencies=["attribute_memory_bottleneck", "diagnose_communication_exposure"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            hints = summary.get("training_hints", {}) or {}
            memory_result = context.get_previous_result("attribute_memory_bottleneck")
            comm_result = context.get_previous_result("diagnose_communication_exposure")

            avg_step = self._to_float(summary.get("avg_step_time"))
            avg_compute = self._to_float(summary.get("avg_compute_time"))
            avg_free = self._to_float(summary.get("avg_free_time"))
            bubble_ratio = self._to_float(summary.get("bubble_ratio"))
            free_ratio = (avg_free / avg_step) if avg_step > 0 else 0.0
            compute_ratio = (avg_compute / avg_step) if avg_step > 0 else 0.0

            memory_code = memory_result.data.get("diagnosis_code", "") if memory_result and memory_result.success else ""
            comm_code = comm_result.data.get("diagnosis_code", "") if comm_result and comm_result.success else ""

            phase = "unknown"
            confidence = 0.35
            evidence: List[str] = []

            if bubble_ratio >= 20 or comm_code == "PIPELINE_BUBBLE_BOUND":
                phase = "pipeline_schedule"
                confidence = 0.85
                evidence.append(f"bubble 占比 {bubble_ratio:.1f}%")
            elif memory_code == "STATE_MEMORY_BOUND":
                phase = "optimizer_step"
                confidence = 0.75
                evidence.append("state-bound 往往在 optimizer.step 或状态常驻阶段最痛")
            elif comm_code == "HSDP_DP_COMM_EXPOSED":
                phase = "grad_state"
                confidence = 0.75
                evidence.append("DP/HSDP 通信通常暴露在 backward 尾部或 grad-state 阶段")
            elif memory_code == "ACTIVATION_MEMORY_BOUND":
                phase = "backward" if hints.get("long_context_likely") or hints.get("activation_heavy_likely") else "forward"
                confidence = 0.70
                evidence.append("activation-bound 通常首先压在 forward/backward 主计算链")
            elif comm_code in {"TP_COMM_EXPOSED", "CP_COMM_EXPOSED", "EP_COMM_EXPOSED"}:
                phase = "forward_backward"
                confidence = 0.65
                evidence.append(f"{comm_code} 更像主计算链上的通信暴露")
            elif free_ratio >= 0.20:
                phase = "input_host_gap"
                confidence = 0.60
                evidence.append(f"空闲占比 {free_ratio * 100:.1f}%")
            elif compute_ratio >= 0.60:
                phase = "forward_backward"
                confidence = 0.50
                evidence.append(f"计算占比 {compute_ratio * 100:.1f}%")

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "dominant_step_phase": phase,
                    "observation_facts": evidence,
                },
                summary=f"Step 归因: {phase}",
                details=evidence,
                evidence=evidence,
                confidence=confidence,
                priority="P1" if phase != "unknown" else "P2",
            )
        except Exception as e:
            logger.exception("Step 阶段归因失败: %s", e)
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
