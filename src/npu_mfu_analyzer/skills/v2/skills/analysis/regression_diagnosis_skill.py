"""性能回归诊断模板技能。"""

import logging

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class RegressionDiagnosisSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="generate_regression_template",
            display_name="性能回归诊断模板",
            description="在有 diff / 双 profiling 输入时生成性能回归排查模板",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.LOW,
            tags=["regression", "ci", "comparison"],
            dependencies=["identify_main_contradiction"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            diff = context.diff_result or context.user_inputs.get("diff_result")
            if not diff:
                return SkillResult(
                    skill_name=self.name,
                    skill_type=self.skill_type,
                    success=True,
                    data={"template": []},
                    summary="无回归输入，跳过回归模板生成",
                    confidence=0.5,
                    priority="P2",
                )

            template = [
                "先比较回归前后 forward/backward/step/save 的时间分解。",
                "确认峰值显存变化落在 activation 还是 optimizer state。",
                "确认通信变化来自 TP/PP/DP 哪一类 collective。",
                "优先用最小 A/B 重现实验验证主矛盾，而不是直接改多层策略。",
            ]
            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={"template": template},
                summary="已生成回归诊断模板",
                details=template,
                recommendations=template,
                confidence=0.75,
                priority="P1",
            )
        except Exception as e:
            logger.exception("回归模板生成失败: %s", e)
            return SkillResult(skill_name=self.name, skill_type=self.skill_type, success=False, error=str(e))
