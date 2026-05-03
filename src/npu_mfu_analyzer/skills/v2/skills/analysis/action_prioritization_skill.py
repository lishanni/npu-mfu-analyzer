"""统一动作优先级技能。"""

import logging
from typing import Dict, List

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class ActionPrioritizationSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="prioritize_optimization_actions",
            display_name="动作优先级收敛",
            description="整合主矛盾、阶段归因、专项场景与 what-if 实验，输出统一动作计划",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.OPTIMIZATION,
            priority=SkillPriority.CRITICAL,
            tags=["optimization", "action-plan", "prioritization"],
            dependencies=[
                "attribute_step_bottleneck",
                "identify_main_contradiction",
                "diagnose_pipeline_behavior",
                "diagnose_training_scenario",
                "diagnose_topology_pressure",
                "suggest_whatif_experiments",
                "generate_regression_template",
            ],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            step_result = context.get_previous_result("attribute_step_bottleneck")
            main_result = context.get_previous_result("identify_main_contradiction")
            pipeline_result = context.get_previous_result("diagnose_pipeline_behavior")
            scenario_result = context.get_previous_result("diagnose_training_scenario")
            topology_result = context.get_previous_result("diagnose_topology_pressure")
            whatif_result = context.get_previous_result("suggest_whatif_experiments")
            regression_result = context.get_previous_result("generate_regression_template")

            main = main_result.data.get("main_contradiction", {}) if main_result and main_result.success else {}
            phase = step_result.data.get("dominant_step_phase", "unknown") if step_result and step_result.success else "unknown"
            scenario = scenario_result.data.get("scenario_code", "GENERAL_DENSE") if scenario_result and scenario_result.success else "GENERAL_DENSE"

            actions: List[Dict[str, str]] = []
            for result in [main_result, pipeline_result, scenario_result, topology_result]:
                if result and result.success:
                    actions.extend(result.data.get("prioritized_actions", result.data.get("candidate_actions", [])))

            deduped: List[Dict[str, str]] = []
            seen = set()
            priority_rank = {"high": 0, "medium": 1, "low": 2}
            actions.sort(key=lambda x: priority_rank.get(str(x.get("priority", "medium")).lower(), 1))
            for action in actions:
                key = action.get("action_code") or action.get("title")
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(action)

            facts: List[str] = []
            for result in [step_result, main_result, pipeline_result, scenario_result, topology_result]:
                if result and result.success:
                    facts.extend(result.data.get("supporting_facts", result.data.get("observation_facts", [])))
            facts = list(dict.fromkeys([f for f in facts if f]))

            experiments = whatif_result.data.get("experiments", []) if whatif_result and whatif_result.success else []
            regression_template = regression_result.data.get("template", []) if regression_result and regression_result.success else []

            plan = {
                "main_contradiction": main,
                "phase_focus": phase,
                "training_scenario": scenario,
                "prioritized_actions": deduped[:6],
                "supporting_facts": facts[:12],
                "experiments": experiments[:4],
                "regression_template": regression_template,
            }

            recommendations = [a.get("title", a.get("action_code", "")) for a in deduped[:6]]
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data=plan,
                summary=f"动作计划: {main.get('code', 'INSUFFICIENT_EVIDENCE')} -> {phase}",
                details=facts[:12],
                recommendations=recommendations,
                actions=deduped[:6],
                root_cause=main.get("reason", ""),
                evidence=facts[:12],
                confidence=float(main.get("confidence", 0.5) or 0.5),
                priority="P0" if main.get("code") else "P2",
                severity="high" if float(main.get("confidence", 0.0) or 0.0) >= 0.75 else "medium",
            )
        except Exception as e:
            logger.exception("动作优先级收敛失败: %s", e)
            return SkillResult(skill_name=self.name, skill_type=self.skill_type, success=False, error=str(e))
