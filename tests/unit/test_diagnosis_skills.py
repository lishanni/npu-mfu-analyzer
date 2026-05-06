import pytest

from npu_mfu_analyzer.agents.base_agent import AnalysisResult
from npu_mfu_analyzer.agents.advisor_agent import AdvisorAgent
from npu_mfu_analyzer.llm.llm_interface import LLMConfig, LLMFactory, LLMInterface, LLMResponse
from npu_mfu_analyzer.skills.v2.base import SkillContext
from npu_mfu_analyzer.skills.v2.engine import SkillEngine
from npu_mfu_analyzer.skills.v2.registry import SkillRegistry
from npu_mfu_analyzer.skills.v2.skills.analysis import (
    ActionPrioritizationSkill,
    CommunicationExposureSkill,
    MainContradictionSkill,
    MemoryAttributionSkill,
    PipelineDiagnosisSkill,
    RegressionDiagnosisSkill,
    ScenarioDiagnosisSkill,
    StepAttributionSkill,
    TopologyDiagnosisSkill,
    WhatIfExperimentSkill,
)


def _build_engine():
    registry = SkillRegistry()
    registry.register(MemoryAttributionSkill())
    registry.register(CommunicationExposureSkill())
    registry.register(StepAttributionSkill())
    registry.register(PipelineDiagnosisSkill())
    registry.register(ScenarioDiagnosisSkill())
    registry.register(TopologyDiagnosisSkill())
    registry.register(WhatIfExperimentSkill())
    registry.register(RegressionDiagnosisSkill())
    registry.register(MainContradictionSkill())
    registry.register(ActionPrioritizationSkill())
    return SkillEngine(registry)


def test_activation_memory_bound_action_plan():
    engine = _build_engine()
    context = SkillContext(
        profiling_summary={
            "avg_step_time": 1000.0,
            "avg_compute_time": 700.0,
            "avg_comm_time": 120.0,
            "activation_memory_mb": 46000,
            "gradient_memory_mb": 10000,
            "optimizer_memory_mb": 4000,
            "model_memory_mb": 6000,
            "peak_memory_mb": 64000,
            "training_hints": {
                "activation_pressure_score": 0.85,
                "state_pressure_score": 0.20,
                "activation_heavy_likely": True,
            },
        },
        user_inputs={
            "agent_results": {
                "memory": AnalysisResult(
                    agent_name="memory",
                    success=True,
                    summary="",
                    details={
                        "peak_memory_gb": 62.5,
                        "peak_memory_mb": 64000,
                        "memory_utilization": 96.0,
                        "oom_risk": "high",
                        "activation_memory_mb": 46000,
                        "optimizer_memory_mb": 4000,
                        "model_memory_mb": 6000,
                        "gradient_memory_mb": 10000,
                        "training_hints": {
                            "activation_pressure_score": 0.85,
                            "state_pressure_score": 0.20,
                            "activation_heavy_likely": True,
                        },
                    },
                ),
                "timeline": AnalysisResult(
                    agent_name="timeline",
                    success=True,
                    summary="",
                    details={"overlap_ratio": 68.0, "bubble_ratio": 4.0},
                ),
                "communication": AnalysisResult(
                    agent_name="communication",
                    success=True,
                    summary="",
                    details={"comm_ratio": 0.12},
                ),
            }
        },
    )
    results = engine.execute_skills(["prioritize_optimization_actions"], context)
    plan = results["prioritize_optimization_actions"].data
    assert plan["main_contradiction"]["code"] == "ACTIVATION_MEMORY_BOUND"
    assert plan["phase_focus"] in {"forward", "backward"}
    assert plan["prioritized_actions"][0]["action_code"] == "enable_recompute"


def test_state_memory_bound_action_plan():
    engine = _build_engine()
    context = SkillContext(
        profiling_summary={
            "avg_step_time": 1200.0,
            "avg_compute_time": 450.0,
            "avg_comm_time": 180.0,
            "activation_memory_mb": 6000,
            "gradient_memory_mb": 4000,
            "optimizer_memory_mb": 36000,
            "model_memory_mb": 18000,
            "peak_memory_mb": 70000,
            "training_hints": {
                "activation_pressure_score": 0.15,
                "state_pressure_score": 0.85,
                "optimizer_heavy_likely": True,
            },
        },
        user_inputs={
            "agent_results": {
                "memory": AnalysisResult(
                    agent_name="memory",
                    success=True,
                    summary="",
                    details={
                        "peak_memory_gb": 68.0,
                        "peak_memory_mb": 70000,
                        "memory_utilization": 92.0,
                        "oom_risk": "high",
                        "activation_memory_mb": 6000,
                        "optimizer_memory_mb": 36000,
                        "model_memory_mb": 18000,
                        "gradient_memory_mb": 4000,
                        "training_hints": {
                            "activation_pressure_score": 0.15,
                            "state_pressure_score": 0.85,
                            "optimizer_heavy_likely": True,
                        },
                    },
                ),
                "timeline": AnalysisResult(
                    agent_name="timeline",
                    success=True,
                    summary="",
                    details={"overlap_ratio": 72.0, "bubble_ratio": 0.0},
                ),
                "communication": AnalysisResult(
                    agent_name="communication",
                    success=True,
                    summary="",
                    details={"comm_ratio": 0.15},
                ),
            }
        },
    )
    results = engine.execute_skills(["prioritize_optimization_actions"], context)
    plan = results["prioritize_optimization_actions"].data
    assert plan["main_contradiction"]["code"] == "STATE_MEMORY_BOUND"
    assert plan["phase_focus"] == "optimizer_step"
    assert plan["prioritized_actions"][0]["action_code"] == "enable_optimizer_state_offload"


def test_pipeline_long_context_and_topology_actions():
    engine = _build_engine()
    context = SkillContext(
        profiling_summary={
            "avg_step_time": 1000.0,
            "avg_compute_time": 420.0,
            "avg_comm_time": 380.0,
            "bubble_ratio": 28.0,
            "rank_count": 16,
            "world_size": 16,
            "overlap_metrics": {"overlap_ratio": 32.0},
            "training_hints": {
                "attention_ratio": 0.42,
                "activation_pressure_score": 0.75,
                "state_pressure_score": 0.20,
                "long_context_likely": True,
                "activation_heavy_likely": True,
            },
        },
        user_inputs={
            "tp_size": 8,
            "pp_size": 2,
            "dp_size": 1,
            "topology_summary": {
                "num_machines": 4,
                "inter_node_ratio": 0.45,
                "slow_links_count": 2,
            },
            "agent_results": {
                "memory": AnalysisResult(
                    agent_name="memory",
                    success=True,
                    summary="",
                    details={
                        "peak_memory_gb": 63.0,
                        "peak_memory_mb": 65000,
                        "memory_utilization": 94.0,
                        "oom_risk": "high",
                        "training_hints": {
                            "attention_ratio": 0.42,
                            "activation_pressure_score": 0.75,
                            "state_pressure_score": 0.20,
                            "long_context_likely": True,
                            "activation_heavy_likely": True,
                        },
                    },
                ),
                "timeline": AnalysisResult(
                    agent_name="timeline",
                    success=True,
                    summary="",
                    details={"overlap_ratio": 32.0, "bubble_ratio": 28.0},
                ),
                "communication": AnalysisResult(
                    agent_name="communication",
                    success=True,
                    summary="",
                    details={
                        "comm_ratio": 0.38,
                        "tp_comm_time_ms": 120.0,
                        "pp_comm_time_ms": 90.0,
                        "dp_comm_time_ms": 20.0,
                        "tp_size": 8,
                        "pp_size": 2,
                        "dp_size": 1,
                        "slow_ranks": [],
                    },
                ),
            },
        },
    )
    results = engine.execute_skills(["prioritize_optimization_actions"], context)
    plan = results["prioritize_optimization_actions"].data
    action_codes = [a["action_code"] for a in plan["prioritized_actions"]]
    assert plan["main_contradiction"]["code"] == "PIPELINE_BUBBLE_BOUND"
    assert plan["training_scenario"] in {"LONG_CONTEXT_DENSE", "PIPELINE_DENSE"}
    assert "tune_pipeline_microbatch" in action_codes
    assert "place_tp_intra_node" in action_codes
    assert plan["experiments"]


def test_moe_scenario_prioritizes_ep_balance():
    engine = _build_engine()
    context = SkillContext(
        profiling_summary={
            "avg_step_time": 1300.0,
            "avg_compute_time": 500.0,
            "avg_comm_time": 340.0,
            "rank_count": 16,
            "world_size": 16,
            "training_hints": {
                "moe_ratio": 0.55,
                "activation_pressure_score": 0.35,
                "state_pressure_score": 0.82,
                "moe_likely": True,
                "optimizer_heavy_likely": True,
            },
        },
        user_inputs={
            "agent_results": {
                "memory": AnalysisResult(
                    agent_name="memory",
                    success=True,
                    summary="",
                    details={
                        "peak_memory_gb": 60.0,
                        "peak_memory_mb": 61440,
                        "memory_utilization": 88.0,
                        "oom_risk": "medium",
                        "optimizer_memory_mb": 42000,
                        "training_hints": {
                            "moe_ratio": 0.55,
                            "activation_pressure_score": 0.35,
                            "state_pressure_score": 0.82,
                            "moe_likely": True,
                            "optimizer_heavy_likely": True,
                        },
                    },
                ),
                "timeline": AnalysisResult(
                    agent_name="timeline",
                    success=True,
                    summary="",
                    details={"overlap_ratio": 40.0, "bubble_ratio": 2.0},
                ),
                "communication": AnalysisResult(
                    agent_name="communication",
                    success=True,
                    summary="",
                    details={
                        "comm_ratio": 0.31,
                        "tp_comm_time_ms": 40.0,
                        "pp_comm_time_ms": 10.0,
                        "dp_comm_time_ms": 20.0,
                        "slow_ranks": [3, 7],
                    },
                ),
            }
        },
    )
    results = engine.execute_skills(["prioritize_optimization_actions"], context)
    plan = results["prioritize_optimization_actions"].data
    action_codes = [a["action_code"] for a in plan["prioritized_actions"]]
    assert plan["training_scenario"] == "MOE_TRAINING"
    assert "tune_ep_load_balance" in action_codes


@pytest.mark.asyncio
async def test_advisor_agent_includes_phase2_to_4_outputs():
    llm = LLMFactory.create(LLMConfig(backend="mock"))
    advisor = AdvisorAgent(llm)
    data = {
        "profiling_summary": {
            "avg_step_time": 1000.0,
            "avg_compute_time": 700.0,
            "avg_comm_time": 120.0,
            "avg_free_time": 80.0,
            "activation_memory_mb": 46000,
            "gradient_memory_mb": 10000,
            "optimizer_memory_mb": 4000,
            "model_memory_mb": 6000,
            "peak_memory_mb": 64000,
            "rank_count": 8,
            "world_size": 8,
            "training_hints": {
                "activation_pressure_score": 0.85,
                "state_pressure_score": 0.20,
                "activation_heavy_likely": True,
            },
        },
        "tp_size": 4,
        "pp_size": 2,
        "dp_size": 1,
        "topology_summary": {
            "num_machines": 2,
            "inter_node_ratio": 0.25,
            "slow_links_count": 1,
        },
        "diff_result": {"baseline": "a", "candidate": "b"},
        "agent_results": {
            "memory": AnalysisResult(
                agent_name="memory",
                success=True,
                summary="",
                details={
                    "peak_memory_gb": 62.5,
                    "peak_memory_mb": 64000,
                    "memory_utilization": 96.0,
                    "oom_risk": "high",
                    "activation_memory_mb": 46000,
                    "optimizer_memory_mb": 4000,
                    "model_memory_mb": 6000,
                    "gradient_memory_mb": 10000,
                    "training_hints": {
                        "activation_pressure_score": 0.85,
                        "state_pressure_score": 0.20,
                        "activation_heavy_likely": True,
                    },
                },
            ),
            "timeline": AnalysisResult(
                agent_name="timeline",
                success=True,
                summary="",
                details={"overlap_ratio": 68.0, "bubble_ratio": 4.0},
            ),
            "communication": AnalysisResult(
                agent_name="communication",
                success=True,
                summary="",
                details={"comm_ratio": 0.12, "tp_size": 4, "pp_size": 2, "dp_size": 1},
            ),
        },
    }
    result = await advisor.analyze(data)
    assert result.success
    diagnosis = result.details["diagnosis"]
    assert diagnosis["main_contradiction"]["code"] == "ACTIVATION_MEMORY_BOUND"
    assert diagnosis["phase_focus"] in {"forward", "backward"}
    assert diagnosis["experiments"]
    assert diagnosis["regression_template"]
    assert result.details["advisor_report"] is not None


class FailingLLM(LLMInterface):
    async def complete(self, messages, **kwargs) -> LLMResponse:
        raise RuntimeError("forced llm failure")


@pytest.mark.asyncio
async def test_advisor_agent_falls_back_to_structured_diagnosis_when_llm_fails():
    advisor = AdvisorAgent(FailingLLM(LLMConfig(backend="mock")))
    data = {
        "profiling_summary": {
            "avg_step_time": 1000.0,
            "avg_compute_time": 700.0,
            "avg_comm_time": 120.0,
            "avg_free_time": 80.0,
            "activation_memory_mb": 46000,
            "gradient_memory_mb": 10000,
            "optimizer_memory_mb": 4000,
            "model_memory_mb": 6000,
            "peak_memory_mb": 64000,
            "rank_count": 8,
            "world_size": 8,
            "training_hints": {
                "activation_pressure_score": 0.85,
                "state_pressure_score": 0.20,
                "activation_heavy_likely": True,
            },
        },
        "tp_size": 4,
        "pp_size": 2,
        "dp_size": 1,
        "topology_summary": {
            "num_machines": 2,
            "inter_node_ratio": 0.25,
            "slow_links_count": 1,
        },
        "agent_results": {
            "memory": AnalysisResult(
                agent_name="memory",
                success=True,
                summary="",
                details={
                    "peak_memory_gb": 62.5,
                    "peak_memory_mb": 64000,
                    "memory_utilization": 96.0,
                    "oom_risk": "high",
                    "activation_memory_mb": 46000,
                    "optimizer_memory_mb": 4000,
                    "model_memory_mb": 6000,
                    "gradient_memory_mb": 10000,
                    "training_hints": {
                        "activation_pressure_score": 0.85,
                        "state_pressure_score": 0.20,
                        "activation_heavy_likely": True,
                    },
                },
            ),
            "timeline": AnalysisResult(
                agent_name="timeline",
                success=True,
                summary="",
                details={"overlap_ratio": 68.0, "bubble_ratio": 4.0},
            ),
            "communication": AnalysisResult(
                agent_name="communication",
                success=True,
                summary="",
                details={"comm_ratio": 0.12, "tp_size": 4, "pp_size": 2, "dp_size": 1},
            ),
        },
    }
    result = await advisor.analyze(data)
    assert result.success
    assert result.details["llm_fallback"] is True
    assert result.details["diagnosis"]["main_contradiction"]["code"] == "ACTIVATION_MEMORY_BOUND"
    assert "主矛盾" in result.raw_response
    assert "优先调优动作" in result.raw_response
