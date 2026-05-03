"""What-if 与实验建议技能。"""

import logging
from typing import Any, Dict, List

from npu_mfu_analyzer.roofline.whatif_simulator import CurrentState, WhatIfSimulator
from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class WhatIfExperimentSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="suggest_whatif_experiments",
            display_name="What-if / A-B 实验建议",
            description="把主矛盾翻译成可验证的下一步实验计划",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.OPTIMIZATION,
            priority=SkillPriority.NORMAL,
            tags=["whatif", "experiment", "ab-test"],
            dependencies=["identify_main_contradiction", "diagnose_training_scenario", "diagnose_communication_exposure"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            main_result = context.get_previous_result("identify_main_contradiction")
            scenario_result = context.get_previous_result("diagnose_training_scenario")
            comm_result = context.get_previous_result("diagnose_communication_exposure")
            main_code = main_result.data.get("main_contradiction", {}).get("code", "") if main_result and main_result.success else ""
            scenario = scenario_result.data.get("scenario_code", "GENERAL_DENSE") if scenario_result and scenario_result.success else "GENERAL_DENSE"
            dominant = comm_result.data.get("dominant_domain", "generic") if comm_result and comm_result.success else "generic"

            world_size = int(summary.get("rank_count", summary.get("world_size", 1)) or 1)
            tp = int(context.user_inputs.get("tp_size") or 1)
            pp = int(context.user_inputs.get("pp_size") or 1)
            dp = int(context.user_inputs.get("dp_size") or max(world_size // max(tp * pp, 1), 1))
            step_time_ms = self._to_float(summary.get("avg_step_time")) / 1000
            mfu_percent = min(95.0, max(5.0, self._to_float(summary.get("avg_compute_time")) / max(self._to_float(summary.get("avg_step_time")), 1.0) * 80.0))
            overlap_ratio = self._to_float((summary.get("overlap_metrics") or {}).get("overlap_ratio")) / 100.0
            comm_time_ratio = self._to_float(summary.get("comm_ratio"))

            simulator = WhatIfSimulator(CurrentState(
                num_devices=world_size,
                tp_size=tp,
                pp_size=pp,
                dp_size=dp,
                step_time_ms=step_time_ms or 1000.0,
                mfu_percent=mfu_percent,
                overlap_ratio=overlap_ratio,
                comm_time_ratio=comm_time_ratio,
            ))

            experiments: List[Dict[str, Any]] = []
            if main_code == "ACTIVATION_MEMORY_BOUND":
                experiments.extend([
                    {"title": "A/B: recompute on/off", "goal": "验证 activation 是否真是主峰", "expected_signal": "峰值显存明显下降，backward 时间上升"},
                    {"title": "A/B: seq_len 降一档", "goal": "验证是否受长序列影响", "expected_signal": "forward/backward 峰值和时间同步下降"},
                ])
            elif main_code == "STATE_MEMORY_BOUND":
                experiments.extend([
                    {"title": "A/B: optimizer-state-offload on/off", "goal": "验证 step 峰值是否由 optimizer state 主导", "expected_signal": "step 峰值下降，step 时间可能上升"},
                    {"title": "A/B: parameter offload / 参数驻留策略", "goal": "区分状态与参数常驻问题", "expected_signal": "显存基线下降，forward 可能更敏感"},
                ])
            elif main_code == "PIPELINE_BUBBLE_BOUND":
                sim = simulator.simulate_parallel_change(new_pp=max(2, pp * 2))
                experiments.append({"title": sim.name, "goal": "评估更细 pipeline 深度的理论收益", "expected_signal": f"模拟 speedup {sim.predicted_speedup:.2f}x", "confidence": sim.confidence})
                experiments.append({"title": "A/B: microbatch 数翻倍", "goal": "验证 bubble 是否主导", "expected_signal": "bubble 比例下降，稳态区间变长"})
            elif dominant == "tp":
                new_tp = max(1, tp // 2) if tp > 1 else min(max(world_size, 1), 2)
                sim = simulator.simulate_parallel_change(new_tp=new_tp)
                experiments.append({"title": sim.name, "goal": "评估收缩/重排 TP 后的理论收益", "expected_signal": f"模拟 speedup {sim.predicted_speedup:.2f}x", "confidence": sim.confidence})
                experiments.append({"title": "A/B: SequenceParallel on/off", "goal": "验证 activation gather 是否是二级矛盾", "expected_signal": "TP 通信暴露和 activation 峰值同时变化"})
            elif scenario == "MOE_TRAINING":
                experiments.extend([
                    {"title": "A/B: expert load balance / capacity 调整", "goal": "验证 router 热点是否拖慢训练", "expected_signal": "最慢 rank 收敛，all-to-all 暴露下降"},
                    {"title": "A/B: optimizer-state-offload on/off", "goal": "验证 expert state 是否主导 step 阶段", "expected_signal": "step 峰值下降"},
                ])
            else:
                experiments.append({"title": "A/B: overlap/prefetch 调整", "goal": "确认是否存在 wait 提前问题", "expected_signal": "纯等待区间缩短"})

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={"experiments": experiments[:4]},
                summary=f"建议 {min(len(experiments), 4)} 组下一步实验",
                details=[e.get("title", "") for e in experiments[:4]],
                recommendations=[e.get("title", "") for e in experiments[:4]],
                confidence=0.7,
                priority="P1",
            )
        except Exception as e:
            logger.exception("What-if/实验建议失败: %s", e)
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
