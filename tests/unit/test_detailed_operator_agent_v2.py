"""
Detailed Operator Agent V2 单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.detailed_operator_agent_v2 import (
    DetailedOperatorAgentV2,
    InstructionMixAnalysis,
    MemoryHierarchyAnalysis,
    StallAnalysis,
    OperatorDeepAnalysisResult,
    OPTIMIZATION_STRATEGIES,
)
from src.data_loader.aic_metrics import (
    ExtendedAICMetrics,
    ExtendedArithmeticUtilization,
    ExtendedMemoryMetrics,
    ExtendedPipelineMetrics,
    DeepBottleneckAnalysis,
    BottleneckCategory,
)


class TestInstructionMixAnalysis:
    """InstructionMixAnalysis 测试"""

    def test_init(self):
        """测试初始化"""
        analysis = InstructionMixAnalysis(
            pattern="cube_dominant",
            cube_utilization=80.0,
            vector_utilization=20.0,
        )
        assert analysis.pattern == "cube_dominant"
        assert analysis.cube_utilization == 80.0
        assert analysis.issues == []


class TestMemoryHierarchyAnalysis:
    """MemoryHierarchyAnalysis 测试"""

    def test_init(self):
        """测试初始化"""
        analysis = MemoryHierarchyAnalysis(
            bottleneck_type="l2_miss",
            l2_hit_rate=30.0,
            ub_utilization=70.0,
        )
        assert analysis.bottleneck_type == "l2_miss"
        assert analysis.l2_hit_rate == 30.0
        assert analysis.ub_utilization == 70.0


class TestStallAnalysis:
    """StallAnalysis 测试"""

    def test_init(self):
        """测试初始化"""
        analysis = StallAnalysis(
            primary_cause="mte",
            primary_rate=40.0,
            total_stall_rate=50.0,
            severity="high",
        )
        assert analysis.primary_cause == "mte"
        assert analysis.primary_rate == 40.0
        assert analysis.severity == "high"

    def test_stall_breakdown(self):
        """测试停顿分解"""
        analysis = StallAnalysis(
            primary_cause="mte",
            stall_breakdown={
                "mte": 40.0,
                "dependency": 20.0,
                "memory": 10.0,
            },
        )
        assert analysis.stall_breakdown["mte"] == 40.0
        assert analysis.stall_breakdown["dependency"] == 20.0


class TestDetailedOperatorAgentV2:
    """DetailedOperatorAgentV2 测试"""

    @pytest.fixture
    def mock_llm(self):
        """创建 mock LLM"""
        llm = MagicMock()
        llm.complete = AsyncMock(return_value="分析结果")
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """创建 Agent 实例"""
        return DetailedOperatorAgentV2(llm=mock_llm)

    def test_init(self, agent):
        """测试初始化"""
        assert agent.name == "DetailedOperatorAgentV2"

    def test_analyze_instruction_mix_cube_dominant(self, agent):
        """测试指令混合分析 - Cube 主导"""
        metrics = ExtendedAICMetrics(
            op_name="MatMul",
            op_type="compute",
            extended_arithmetic=ExtendedArithmeticUtilization(
                cube_utilization=80.0,
                vector_utilization=20.0,
                scalar_utilization=5.0,
            ),
        )

        analysis = agent._analyze_instruction_mix(metrics)

        assert analysis.pattern == "cube_dominant"
        assert analysis.cube_utilization == 80.0

    def test_analyze_instruction_mix_vector_dominant(self, agent):
        """测试指令混合分析 - Vector 主导"""
        metrics = ExtendedAICMetrics(
            op_name="Add",
            op_type="elementwise",
            extended_arithmetic=ExtendedArithmeticUtilization(
                cube_utilization=10.0,
                vector_utilization=70.0,
                scalar_utilization=5.0,
            ),
        )

        analysis = agent._analyze_instruction_mix(metrics)

        assert analysis.pattern == "vector_dominant"
        assert analysis.vector_utilization == 70.0

    def test_analyze_instruction_mix_balanced(self, agent):
        """测试指令混合分析 - 均衡"""
        metrics = ExtendedAICMetrics(
            op_name="MixedOp",
            op_type="mixed",
            extended_arithmetic=ExtendedArithmeticUtilization(
                cube_utilization=45.0,
                vector_utilization=40.0,
                scalar_utilization=15.0,
            ),
        )

        analysis = agent._analyze_instruction_mix(metrics)

        assert analysis.pattern == "balanced"

    def test_analyze_instruction_mix_issues(self, agent):
        """测试指令混合分析 - 问题检测"""
        metrics = ExtendedAICMetrics(
            op_name="LowUtil",
            op_type="compute",
            extended_arithmetic=ExtendedArithmeticUtilization(
                cube_utilization=15.0,  # 低利用率
                vector_utilization=20.0,
                instruction_issue_rate=40.0,  # 低发射率
            ),
        )

        analysis = agent._analyze_instruction_mix(metrics)

        assert "cube_underutilized" in analysis.issues
        assert "low_issue_rate" in analysis.issues

    def test_analyze_memory_hierarchy_normal(self, agent):
        """测试内存层次分析 - 正常"""
        metrics = ExtendedAICMetrics(
            op_name="MatMul",
            op_type="compute",
            extended_memory=ExtendedMemoryMetrics(
                l2_cache_hit_rate=85.0,
                ub_usage=50.0,
                l0_usage=40.0,
            ),
        )

        analysis = agent._analyze_memory_hierarchy(metrics)

        assert analysis.bottleneck_type == "none"
        assert analysis.l2_hit_rate == 85.0

    def test_analyze_memory_hierarchy_l2_miss(self, agent):
        """测试内存层次分析 - L2 命中率低"""
        metrics = ExtendedAICMetrics(
            op_name="LargeMatMul",
            op_type="compute",
            extended_memory=ExtendedMemoryMetrics(
                l2_cache_hit_rate=35.0,  # 低命中率
                ub_usage=50.0,
            ),
        )

        analysis = agent._analyze_memory_hierarchy(metrics)

        assert analysis.bottleneck_type == "l2_miss"
        assert "l2_miss" in analysis.issues

    def test_analyze_memory_hierarchy_ub_pressure(self, agent):
        """测试内存层次分析 - UB 压力大"""
        metrics = ExtendedAICMetrics(
            op_name="LargeTensor",
            op_type="compute",
            extended_memory=ExtendedMemoryMetrics(
                l2_cache_hit_rate=75.0,
                ub_usage=90.0,  # UB 使用率高
            ),
        )

        analysis = agent._analyze_memory_hierarchy(metrics)

        assert analysis.bottleneck_type == "ub_pressure"
        assert "ub_pressure" in analysis.issues

    def test_analyze_stall_reasons_normal(self, agent):
        """测试停顿分析 - 正常"""
        metrics = ExtendedAICMetrics(
            op_name="MatMul",
            op_type="compute",
            extended_pipeline=ExtendedPipelineMetrics(
                stall_rate=10.0,
                pipe_utilization=85.0,
            ),
        )

        analysis = agent._analyze_stall_reasons(metrics)

        assert analysis.severity == "low"
        assert analysis.total_stall_rate == 10.0

    def test_analyze_stall_reasons_critical(self, agent):
        """测试停顿分析 - 严重"""
        metrics = ExtendedAICMetrics(
            op_name="BadOp",
            op_type="compute",
            extended_pipeline=ExtendedPipelineMetrics(
                stall_rate=60.0,
                pipe_utilization=30.0,
                mte_stall_rate=40.0,
                dependency_stall_rate=15.0,
            ),
        )

        analysis = agent._analyze_stall_reasons(metrics)

        assert analysis.severity == "critical"
        assert analysis.primary_cause == "mte"

    def test_identify_deep_bottleneck_compute(self, agent):
        """测试深度瓶颈识别 - 计算瓶颈"""
        instruction = InstructionMixAnalysis(
            pattern="balanced",
            issues=["cube_underutilized"],
            cube_utilization=15.0,
        )
        memory = MemoryHierarchyAnalysis(
            bottleneck_type="none",
            l2_hit_rate=80.0,
        )
        stall = StallAnalysis(
            primary_cause="other",
            total_stall_rate=10.0,
            severity="low",
        )

        bottleneck = agent._identify_deep_bottleneck(instruction, memory, stall)

        assert bottleneck.bottleneck_category == "compute_bound"
        assert bottleneck.compute_detail == "cube_underutilized"

    def test_identify_deep_bottleneck_memory(self, agent):
        """测试深度瓶颈识别 - 内存瓶颈"""
        instruction = InstructionMixAnalysis(
            pattern="balanced",
            issues=[],
            cube_utilization=60.0,
        )
        memory = MemoryHierarchyAnalysis(
            bottleneck_type="l2_miss",
            l2_hit_rate=30.0,
            locality_score=40.0,
        )
        stall = StallAnalysis(
            primary_cause="other",
            total_stall_rate=15.0,
            severity="low",
        )

        bottleneck = agent._identify_deep_bottleneck(instruction, memory, stall)

        assert bottleneck.bottleneck_category == "memory_bound"
        assert bottleneck.memory_detail == "l2_miss"

    def test_identify_deep_bottleneck_pipeline(self, agent):
        """测试深度瓶颈识别 - 流水线瓶颈"""
        instruction = InstructionMixAnalysis(
            pattern="balanced",
            issues=[],
            cube_utilization=60.0,
        )
        memory = MemoryHierarchyAnalysis(
            bottleneck_type="none",
            l2_hit_rate=80.0,
        )
        stall = StallAnalysis(
            primary_cause="mte",
            total_stall_rate=50.0,
            severity="critical",
        )

        bottleneck = agent._identify_deep_bottleneck(instruction, memory, stall)

        assert bottleneck.bottleneck_category == "pipeline_bound"
        assert "mte" in bottleneck.pipeline_detail

    def test_identify_deep_bottleneck_balanced(self, agent):
        """测试深度瓶颈识别 - 均衡"""
        instruction = InstructionMixAnalysis(
            pattern="balanced",
            issues=[],
            cube_utilization=60.0,
        )
        memory = MemoryHierarchyAnalysis(
            bottleneck_type="none",
            l2_hit_rate=80.0,
        )
        stall = StallAnalysis(
            primary_cause="other",
            total_stall_rate=10.0,
            severity="low",
        )

        bottleneck = agent._identify_deep_bottleneck(instruction, memory, stall)

        assert bottleneck.bottleneck_category == "balanced"

    def test_identify_deep_bottleneck_mixed(self, agent):
        """测试深度瓶颈识别 - 混合瓶颈"""
        instruction = InstructionMixAnalysis(
            pattern="balanced",
            issues=["cube_underutilized"],
            cube_utilization=15.0,
        )
        memory = MemoryHierarchyAnalysis(
            bottleneck_type="l2_miss",
            l2_hit_rate=35.0,
        )
        stall = StallAnalysis(
            primary_cause="mte",
            total_stall_rate=35.0,
            severity="high",
        )

        bottleneck = agent._identify_deep_bottleneck(instruction, memory, stall)

        assert bottleneck.bottleneck_category == "mixed"

    def test_generate_targeted_recommendations(self, agent):
        """测试生成针对性优化建议"""
        bottleneck = DeepBottleneckAnalysis(
            bottleneck_category="compute_bound",
            compute_detail="cube_underutilized",
            primary_optimization="优化数据布局以提高计算密度",
            secondary_optimizations=["考虑使用更高维度的矩阵乘法"],
            estimated_speedup=1.3,
        )

        recommendations = agent._generate_targeted_recommendations(bottleneck)

        assert len(recommendations) > 0
        assert "计算密度" in recommendations[0] or "矩阵乘法" in recommendations[1]


class TestOptimizationStrategies:
    """优化策略测试"""

    def test_compute_bound_strategies_exist(self):
        """测试计算瓶颈策略存在"""
        assert "compute_bound" in OPTIMIZATION_STRATEGIES
        assert "cube_underutilized" in OPTIMIZATION_STRATEGIES["compute_bound"]

    def test_memory_bound_strategies_exist(self):
        """测试内存瓶颈策略存在"""
        assert "memory_bound" in OPTIMIZATION_STRATEGIES
        assert "l2_miss" in OPTIMIZATION_STRATEGIES["memory_bound"]

    def test_pipeline_bound_strategies_exist(self):
        """测试流水线瓶颈策略存在"""
        assert "pipeline_bound" in OPTIMIZATION_STRATEGIES
        assert "mte" in OPTIMIZATION_STRATEGIES["pipeline_bound"]

    def test_balanced_strategies_exist(self):
        """测试均衡状态策略存在"""
        assert "balanced" in OPTIMIZATION_STRATEGIES


class TestExtendedAICMetrics:
    """ExtendedAICMetrics 测试"""

    def test_perform_deep_analysis(self):
        """测试执行深度分析"""
        metrics = ExtendedAICMetrics(
            op_name="MatMul",
            op_type="compute",
            extended_arithmetic=ExtendedArithmeticUtilization(
                cube_utilization=15.0,
                vector_utilization=20.0,
            ),
            extended_memory=ExtendedMemoryMetrics(
                l2_cache_hit_rate=80.0,
            ),
            extended_pipeline=ExtendedPipelineMetrics(
                stall_rate=10.0,
            ),
        )

        bottleneck = metrics.perform_deep_analysis()

        assert bottleneck is not None
        assert bottleneck.bottleneck_category == "compute_bound"
        assert bottleneck.compute_detail == "cube_underutilized"

    def test_to_dict(self):
        """测试转换为字典"""
        metrics = ExtendedAICMetrics(
            op_name="MatMul",
            op_type="compute",
            duration_us=1000.0,
            extended_arithmetic=ExtendedArithmeticUtilization(
                cube_utilization=50.0,
            ),
        )

        d = metrics.to_dict()

        assert d["op_name"] == "MatMul"
        assert d["duration_us"] == 1000.0
        assert "extended_arithmetic" in d