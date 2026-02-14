"""
对比分析功能单元测试

测试 SimilarityChecker, ProfilingDiffEngine, ComparisonAdvisorAgent
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.data_loader.data_summarizer import (
    ProfilingSummary,
    OverlapMetrics,
    StepMetrics,
)
from src.data_loader.profiling_loader import ProfilingInfo
from src.analyzers.similarity_checker import (
    SimilarityChecker,
    SimilarityResult,
    ComparabilityLevel,
)
from src.analyzers.profiling_diff import (
    ProfilingDiffEngine,
    ProfilingDiff,
    MetricChange,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def profiling_info_a():
    """模拟 Profiling A 信息"""
    return ProfilingInfo(
        path="/path/to/profiling_a",
        data_type="db",
        framework="pytorch",
        rank_count=8,
        has_timeline=True,
        has_memory=True,
        has_communication=True,
        db_paths=["/path/to/profiling_a/db.db"],
        json_paths=[],
    )


@pytest.fixture
def profiling_info_b():
    """模拟 Profiling B 信息（与 A 相同硬件）"""
    return ProfilingInfo(
        path="/path/to/profiling_b",
        data_type="db",
        framework="pytorch",
        rank_count=8,
        has_timeline=True,
        has_memory=True,
        has_communication=True,
        db_paths=["/path/to/profiling_b/db.db"],
        json_paths=[],
    )


@pytest.fixture
def profiling_info_different():
    """模拟差异很大的 Profiling 信息"""
    return ProfilingInfo(
        path="/path/to/profiling_diff",
        data_type="json",
        framework="mindspore",
        rank_count=64,
        has_timeline=True,
        has_memory=False,
        has_communication=True,
        db_paths=[],
        json_paths=["/path/to/profiling_diff/trace.json"],
    )


@pytest.fixture
def summary_a():
    """模拟 Profiling A 摘要"""
    return ProfilingSummary(
        data_path="/path/to/profiling_a",
        data_type="db",
        framework="pytorch",
        rank_count=8,
        step_count=100,
        avg_step_time=500000,      # 500ms in us
        avg_compute_time=350000,    # 350ms
        avg_comm_time=100000,       # 100ms
        avg_free_time=50000,        # 50ms
        overlap_metrics=OverlapMetrics(
            total_compute_time=350000,
            total_comm_time=100000,
            overlapped_time=60000,
            comm_not_overlapped=40000,
            free_time=50000,
            overlap_ratio=60.0,
        ),
        avg_bubble_time=0,
        bubble_ratio=0,
        top_operators=[
            {"name": "MatMul", "dur": 150000},
            {"name": "AllReduce", "dur": 80000},
            {"name": "LayerNorm", "dur": 30000},
            {"name": "Softmax", "dur": 20000},
            {"name": "Add", "dur": 10000},
        ],
        sample_steps=[
            StepMetrics(step=0, computing=350000, communication=100000, free=50000),
            StepMetrics(step=1, computing=355000, communication=98000, free=47000),
            StepMetrics(step=2, computing=348000, communication=102000, free=50000),
        ],
    )


@pytest.fixture
def summary_b():
    """模拟 Profiling B 摘要（性能稍有劣化）"""
    return ProfilingSummary(
        data_path="/path/to/profiling_b",
        data_type="db",
        framework="pytorch",
        rank_count=8,
        step_count=100,
        avg_step_time=550000,      # 550ms (10% 劣化)
        avg_compute_time=360000,    # 360ms (略增)
        avg_comm_time=130000,       # 130ms (30% 增加)
        avg_free_time=60000,        # 60ms
        overlap_metrics=OverlapMetrics(
            total_compute_time=360000,
            total_comm_time=130000,
            overlapped_time=60000,
            comm_not_overlapped=70000,
            free_time=60000,
            overlap_ratio=46.2,
        ),
        avg_bubble_time=0,
        bubble_ratio=0,
        top_operators=[
            {"name": "MatMul", "dur": 155000},
            {"name": "AllReduce", "dur": 110000},  # 明显增加
            {"name": "LayerNorm", "dur": 32000},
            {"name": "Softmax", "dur": 20000},
            {"name": "Cast", "dur": 15000},          # 新增
        ],
        sample_steps=[
            StepMetrics(step=0, computing=360000, communication=130000, free=60000),
            StepMetrics(step=1, computing=362000, communication=128000, free=60000),
            StepMetrics(step=2, computing=358000, communication=132000, free=60000),
        ],
    )


@pytest.fixture
def summary_improved():
    """模拟性能改善的 Profiling 摘要"""
    return ProfilingSummary(
        data_path="/path/to/profiling_improved",
        data_type="db",
        framework="pytorch",
        rank_count=8,
        step_count=100,
        avg_step_time=420000,      # 420ms (16% 改善)
        avg_compute_time=330000,    # 330ms
        avg_comm_time=60000,        # 60ms (40% 减少)
        avg_free_time=30000,        # 30ms (40% 减少)
        overlap_metrics=OverlapMetrics(
            total_compute_time=330000,
            total_comm_time=60000,
            overlapped_time=45000,
            comm_not_overlapped=15000,
            free_time=30000,
            overlap_ratio=75.0,
        ),
        avg_bubble_time=0,
        bubble_ratio=0,
        top_operators=[
            {"name": "MatMul", "dur": 140000},
            {"name": "AllReduce", "dur": 50000},
            {"name": "LayerNorm", "dur": 28000},
            {"name": "Softmax", "dur": 18000},
            {"name": "Add", "dur": 8000},
        ],
        sample_steps=[
            StepMetrics(step=0, computing=330000, communication=60000, free=30000),
            StepMetrics(step=1, computing=332000, communication=58000, free=30000),
            StepMetrics(step=2, computing=328000, communication=62000, free=30000),
        ],
    )


# ============================================================
# SimilarityChecker Tests
# ============================================================

class TestSimilarityChecker:
    """相似度检测器测试"""

    def test_identical_profiling_high_similarity(
        self, profiling_info_a, profiling_info_b, summary_a, summary_b
    ):
        """相同配置的 Profiling 应该有高相似度"""
        checker = SimilarityChecker()
        result = checker.check(
            profiling_info_a, profiling_info_b,
            summary_a, summary_b,
        )

        assert result.overall_score >= 0.5
        assert result.level in (
            ComparabilityLevel.COMPARABLE,
            ComparabilityLevel.PARTIALLY_COMPARABLE,
        )
        assert result.is_comparable()

    def test_different_profiling_low_similarity(
        self, profiling_info_a, profiling_info_different, summary_a
    ):
        """差异很大的 Profiling 应该有低相似度"""
        # 创建一个差异很大的 summary
        different_summary = ProfilingSummary(
            data_path="/path/to/different",
            data_type="json",
            framework="mindspore",
            rank_count=64,
            step_count=5,
            avg_step_time=5000000,  # 5000ms vs 500ms (10x)
            avg_compute_time=100000,
            avg_comm_time=4800000,  # 完全不同的时间分布
            avg_free_time=100000,
            top_operators=[
                {"name": "TransData", "dur": 2000000},
                {"name": "Matmul_v2", "dur": 100000},
            ],
        )

        checker = SimilarityChecker()
        result = checker.check(
            profiling_info_a, profiling_info_different,
            summary_a, different_summary,
        )

        # 不同框架、不同卡数、不同算子 -> 低相似度
        assert result.overall_score < 0.6
        assert len(result.warnings) > 0

    def test_same_operators_boost_similarity(
        self, profiling_info_a, profiling_info_b, summary_a, summary_b
    ):
        """共同算子多应该提高相似度"""
        ops_a = [
            {"name": "MatMul"}, {"name": "AllReduce"},
            {"name": "LayerNorm"}, {"name": "Softmax"},
        ]
        ops_b = [
            {"name": "MatMul"}, {"name": "AllReduce"},
            {"name": "LayerNorm"}, {"name": "Softmax"},
        ]

        checker = SimilarityChecker()
        result = checker.check(
            profiling_info_a, profiling_info_b,
            summary_a, summary_b,
            operators_a=ops_a,
            operators_b=ops_b,
        )

        assert result.overall_score >= 0.6

    def test_similarity_result_to_markdown(
        self, profiling_info_a, profiling_info_b, summary_a, summary_b
    ):
        """相似度结果转 Markdown 不应报错"""
        checker = SimilarityChecker()
        result = checker.check(
            profiling_info_a, profiling_info_b,
            summary_a, summary_b,
        )

        md = result.to_markdown()
        assert "相似度评估" in md
        assert "总体评分" in md

    def test_similarity_result_to_dict(
        self, profiling_info_a, profiling_info_b, summary_a, summary_b
    ):
        """相似度结果转 dict 不应报错"""
        checker = SimilarityChecker()
        result = checker.check(
            profiling_info_a, profiling_info_b,
            summary_a, summary_b,
        )

        d = result.to_dict()
        assert "overall_score" in d
        assert "level" in d
        assert "dimensions" in d
        assert len(d["dimensions"]) == 4


# ============================================================
# ProfilingDiffEngine Tests
# ============================================================

class TestProfilingDiffEngine:
    """差异分析引擎测试"""

    def test_summary_diff_degraded(self, summary_a, summary_b):
        """性能劣化场景的 Summary Diff"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        assert diff.summary_diff is not None
        assert diff.overall_verdict in ("degraded", "mixed")

        # step time 应该增加
        assert diff.summary_diff.step_time is not None
        assert diff.summary_diff.step_time.change_pct > 0  # 变慢
        assert not diff.summary_diff.step_time.is_improvement

    def test_summary_diff_improved(self, summary_a, summary_improved):
        """性能改善场景的 Summary Diff"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_improved)

        assert diff.summary_diff is not None
        assert diff.overall_verdict == "improved"

        # step time 应该减少
        assert diff.summary_diff.step_time is not None
        assert diff.summary_diff.step_time.change_pct < 0
        assert diff.summary_diff.step_time.is_improvement

    def test_operator_diff(self, summary_a, summary_b):
        """算子级差异"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        od = diff.operator_diff
        assert od is not None

        # AllReduce 应该在劣化列表中（80000 -> 110000）
        regression_names = [op.name for op in od.top_regressions]
        assert "AllReduce" in regression_names

        # "Add" 消失，"Cast" 新增
        new_names = [op.name for op in od.new_operators]
        removed_names = [op.name for op in od.removed_operators]
        assert "Cast" in new_names
        assert "Add" in removed_names

    def test_timeline_diff(self, summary_a, summary_b):
        """Timeline 级差异"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        td = diff.timeline_diff
        assert td is not None
        # 应该有 step_time_cv 的计算结果
        assert td.step_time_cv_a >= 0
        assert td.step_time_cv_b >= 0

    def test_comm_diff(self, summary_a, summary_b):
        """通信级差异"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        cd = diff.comm_diff
        assert cd is not None
        assert cd.total_comm_time_change is not None
        # 通信时间应该增加 (100000 -> 130000)
        assert cd.total_comm_time_change.change_pct > 0

    def test_diff_to_prompt_text(self, summary_a, summary_b):
        """差异结果转 Prompt 文本"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        text = diff.to_prompt_text()
        assert "Profiling 差异分析结果" in text
        assert "整体判断" in text

    def test_diff_to_dict(self, summary_a, summary_b):
        """差异结果转 dict"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        d = diff.to_dict()
        assert "overall_verdict" in d
        assert "summary_diff" in d
        assert "primary_changes" in d

    def test_primary_changes_extraction(self, summary_a, summary_b):
        """主要变化提取"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        assert len(diff.primary_changes) > 0
        # Step 时间变化应该在主要变化中
        assert any("Step 时间" in c for c in diff.primary_changes)

    def test_unchanged_profiling(self, summary_a):
        """相同数据对比应该是 unchanged"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_a)

        assert diff.overall_verdict == "unchanged"
        assert diff.summary_diff.improvement_count == 0
        assert diff.summary_diff.regression_count == 0

    def test_metric_change_significance(self):
        """MetricChange 显著性分类"""
        engine = ProfilingDiffEngine()

        # 高显著性
        change_high = engine._make_change(
            "test", "测试", 100, 115, unit="ms", lower_is_better=True
        )
        assert change_high.significance == "high"

        # 中等显著性
        change_med = engine._make_change(
            "test", "测试", 100, 107, unit="ms", lower_is_better=True
        )
        assert change_med.significance == "medium"

        # 低显著性
        change_low = engine._make_change(
            "test", "测试", 100, 102, unit="ms", lower_is_better=True
        )
        assert change_low.significance == "low"


# ============================================================
# ComparisonAdvisorAgent Tests
# ============================================================

class TestComparisonAdvisorAgent:
    """对比分析 Agent 测试"""

    @pytest.mark.asyncio
    async def test_rule_based_analysis(self, summary_a, summary_b):
        """规则引擎分析（不依赖 LLM）"""
        from src.agents.comparison_agent import ComparisonAdvisorAgent
        from src.llm.llm_interface import LLMConfig, LLMFactory

        # 使用 mock LLM
        llm_config = LLMConfig(backend="mock")
        llm = LLMFactory.create(llm_config)

        agent = ComparisonAdvisorAgent(llm)

        # 构造差异数据
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        data = {
            "diff": diff,
            "diff_text": diff.to_prompt_text(),
            "similarity_text": "相似度: 85%",
            "label_a": "基准版本",
            "label_b": "当前版本",
            "path_a": "/path/a",
            "path_b": "/path/b",
            "rank_a": 8,
            "rank_b": 8,
            "step_a": 100,
            "step_b": 100,
        }

        result = await agent.analyze(data)

        assert result.success
        assert result.agent_name == "ComparisonAdvisorAgent"

    def test_rule_based_fallback(self, summary_a, summary_b):
        """规则引擎降级分析"""
        from src.agents.comparison_agent import ComparisonAdvisorAgent
        from src.llm.llm_interface import LLMConfig, LLMFactory

        llm = LLMFactory.create(LLMConfig(backend="mock"))
        agent = ComparisonAdvisorAgent(llm)

        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        insights = agent._rule_based_analysis({"diff": diff})
        assert len(insights) > 0
        # 应该包含通信相关的洞察
        assert any("通信" in i for i in insights)


# ============================================================
# Integration-like Tests
# ============================================================

class TestComparisonIntegration:
    """对比分析集成测试（使用 mock 数据）"""

    def test_full_diff_pipeline(self, summary_a, summary_b):
        """完整的差异分析流水线"""
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        # 验证所有层级都有结果
        assert diff.summary_diff is not None
        assert diff.timeline_diff is not None
        assert diff.operator_diff is not None
        assert diff.comm_diff is not None
        assert diff.memory_diff is not None
        assert diff.overall_verdict != ""
        assert len(diff.primary_changes) > 0

        # 验证序列化
        d = diff.to_dict()
        assert isinstance(d, dict)

        # 验证 Prompt 生成
        text = diff.to_prompt_text()
        assert len(text) > 100

    def test_similarity_then_diff(
        self, profiling_info_a, profiling_info_b, summary_a, summary_b
    ):
        """先检查相似度，再进行差异分析"""
        # Step 1: 相似度检查
        checker = SimilarityChecker()
        similarity = checker.check(
            profiling_info_a, profiling_info_b,
            summary_a, summary_b,
        )

        assert similarity.is_comparable()

        # Step 2: 差异分析
        engine = ProfilingDiffEngine()
        diff = engine.compute(summary_a, summary_b)

        assert diff.overall_verdict in ("improved", "degraded", "mixed", "unchanged")
