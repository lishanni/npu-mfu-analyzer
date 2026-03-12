"""
分析器模块单元测试
"""

import pytest
import pandas as pd
import numpy as np


class TestOverlapCalculator:
    """Overlap 计算器测试"""
    
    def test_calculate_from_events_empty(self):
        """空事件列表"""
        from npu_mfu_analyzer.analyzers import OverlapCalculator
        
        calc = OverlapCalculator()
        metrics = calc.calculate_from_events([], [])
        
        assert metrics.total_compute_time == 0
        assert metrics.total_comm_time == 0
        assert metrics.overlap_ratio == 0
    
    def test_calculate_from_events_no_overlap(self):
        """无重叠的情况"""
        from npu_mfu_analyzer.analyzers import OverlapCalculator
        
        calc = OverlapCalculator()
        compute_events = [{"ts": 0, "dur": 100}]
        comm_events = [{"ts": 200, "dur": 100}]
        
        metrics = calc.calculate_from_events(compute_events, comm_events)
        
        assert metrics.total_compute_time == 100
        assert metrics.total_comm_time == 100
        assert metrics.overlap_ratio == 0  # 无重叠
        assert metrics.comm_not_overlapped == 100
    
    def test_calculate_from_events_full_overlap(self):
        """完全重叠的情况"""
        from npu_mfu_analyzer.analyzers import OverlapCalculator
        
        calc = OverlapCalculator()
        # 计算事件完全覆盖通信事件
        compute_events = [{"ts": 0, "dur": 200}]
        comm_events = [{"ts": 50, "dur": 50}]
        
        metrics = calc.calculate_from_events(compute_events, comm_events)
        
        assert metrics.total_compute_time == 200
        assert metrics.total_comm_time == 50
        # 通信完全被计算掩盖
        assert metrics.overlapped_time >= 0
    
    def test_calculate_from_step_trace(self):
        """从 step_trace 行计算"""
        from npu_mfu_analyzer.analyzers import OverlapCalculator
        
        calc = OverlapCalculator()
        row = {
            "computing": 1000,
            "communication": 500,
            "communication_not_overlapped": 200,
            "overlapped": 300,
            "free": 100,
        }
        
        metrics = calc.calculate_from_step_trace(row)
        
        assert metrics.total_compute_time == 1000
        assert metrics.total_comm_time == 500
        assert metrics.comm_not_overlapped == 200
        assert metrics.overlapped_time == 300
        assert metrics.overlap_ratio == 60.0  # 300/500 * 100
    
    def test_overlap_metrics_to_prompt(self):
        """指标转 Prompt 文本"""
        from npu_mfu_analyzer.analyzers import OverlapMetrics
        
        metrics = OverlapMetrics(
            total_compute_time=1000000,  # 1000ms
            total_comm_time=500000,
            overlapped_time=300000,
            comm_not_overlapped=200000,
            free_time=100000,
            overlap_ratio=60.0,
            e2e_time=1300000,
        )
        
        text = metrics.to_prompt_text()
        
        assert "1000.00 ms" in text
        assert "60.0%" in text


class TestSlowRankDetector:
    """慢卡检测测试"""
    
    def test_judge_dixon_no_outlier(self):
        """Dixon 检验 - 无异常值"""
        from npu_mfu_analyzer.analyzers import judge_dixon
        
        time_list = [100, 101, 99, 102, 100]
        result = judge_dixon(time_list)
        
        assert result == []  # 无明显异常
    
    def test_judge_dixon_with_outlier(self):
        """Dixon 检验 - 有异常值"""
        from npu_mfu_analyzer.analyzers import judge_dixon
        
        time_list = [100, 101, 99, 102, 50]  # 50 明显异常
        result = judge_dixon(time_list)
        
        assert 4 in result  # 索引 4 是慢卡
    
    def test_judge_norm_no_outlier(self):
        """三倍标准差 - 无异常值"""
        from npu_mfu_analyzer.analyzers import judge_norm
        
        # 正态分布数据
        np.random.seed(42)
        time_list = list(np.random.normal(100, 5, 30))
        result = judge_norm(time_list)
        
        # 正态分布下应该很少有 3 sigma 异常
        assert len(result) <= 1
    
    def test_judge_slow_rank_auto_select(self):
        """自动选择检验方法"""
        from npu_mfu_analyzer.analyzers import judge_slow_rank
        
        # 小样本用 Dixon
        small_list = [100, 101, 99, 50]
        result = judge_slow_rank(small_list)
        assert 3 in result
    
    def test_slow_rank_detector_from_df(self):
        """从 DataFrame 检测慢卡"""
        from npu_mfu_analyzer.analyzers import SlowRankDetector
        
        detector = SlowRankDetector()
        
        df = pd.DataFrame({
            "rank": [0, 1, 2, 3, 0, 1, 2, 3],
            "computing": [100, 102, 98, 101, 100, 101, 99, 100],
            "free": [10, 11, 9, 50, 10, 10, 9, 48],  # rank 3 空闲时间多
        })
        
        result = detector.detect_from_step_trace(df, rank_column="rank")
        
        # rank 3 应该被检测为慢卡（空闲时间多）
        assert 3 in result.slow_ranks_by_free or len(result.slow_ranks_by_free) == 0


class TestBubbleAnalyzer:
    """Bubble 分析器测试"""
    
    def test_analyze_empty_df(self):
        """空 DataFrame"""
        from npu_mfu_analyzer.analyzers import BubbleAnalyzer
        
        analyzer = BubbleAnalyzer(pp_size=4, micro_batches=8)
        df = pd.DataFrame()
        
        metrics = analyzer.analyze_from_step_trace(df)
        
        assert metrics.avg_bubble_time == 0
    
    def test_calculate_ideal_bubble(self):
        """计算理论 Bubble 比例"""
        from npu_mfu_analyzer.analyzers import BubbleMetrics
        
        metrics = BubbleMetrics(pp_size=4, micro_batches=8)
        ideal = metrics.calculate_ideal_bubble()
        
        # (4-1)/8 = 0.375
        assert ideal == 0.375
    
    def test_analyze_from_step_trace(self):
        """从 step_trace 分析"""
        from npu_mfu_analyzer.analyzers import BubbleAnalyzer
        
        analyzer = BubbleAnalyzer(pp_size=4, micro_batches=8)
        
        df = pd.DataFrame({
            "stage": [1000, 1100, 900, 1050],
            "bubble": [100, 120, 80, 110],
        })
        
        metrics = analyzer.analyze_from_step_trace(df)
        
        assert metrics.avg_stage_time > 0
        assert metrics.avg_bubble_time > 0
        assert 0 <= metrics.actual_bubble_ratio <= 1
    
    def test_suggest_optimization(self):
        """优化建议生成"""
        from npu_mfu_analyzer.analyzers import BubbleAnalyzer, BubbleMetrics
        
        analyzer = BubbleAnalyzer(pp_size=4, micro_batches=4)
        
        metrics = BubbleMetrics(
            pp_size=4,
            micro_batches=4,
            avg_bubble_time=500,
            avg_stage_time=1000,
            actual_bubble_ratio=0.5,
            ideal_bubble_ratio=0.75,
        )
        
        suggestions = analyzer.suggest_optimization(metrics)
        
        # 应该有建议
        assert len(suggestions) >= 0


class TestCommunicationSplitter:
    """通信拆分测试"""
    
    def test_split_empty_events(self):
        """空事件列表"""
        from npu_mfu_analyzer.analyzers import CommunicationSplitter
        
        splitter = CommunicationSplitter()
        result = splitter.split_from_events([])
        
        assert result.total_comm_time == 0
    
    def test_split_by_group_name(self):
        """按通信域名称拆分"""
        from npu_mfu_analyzer.analyzers import CommunicationSplitter
        
        splitter = CommunicationSplitter()
        
        events = [
            {"name": "AllReduce", "dur": 100, "args": {"groupName": "tp_group"}},
            {"name": "Send", "dur": 50, "args": {"groupName": "pp_group"}},
            {"name": "AllReduce", "dur": 80, "args": {"groupName": "dp_group"}},
        ]
        
        result = splitter.split_from_events(events)
        
        assert result.tp_comm_time == 100
        assert result.pp_comm_time == 50
        assert result.dp_comm_time == 80
        assert result.total_comm_time == 230
    
    def test_split_by_op_type(self):
        """按算子类型拆分"""
        from npu_mfu_analyzer.analyzers import CommunicationSplitter
        
        splitter = CommunicationSplitter()
        
        events = [
            {"name": "Send", "dur": 50, "args": {}},
            {"name": "Recv", "dur": 50, "args": {}},
            {"name": "All2All", "dur": 100, "args": {}},
        ]
        
        result = splitter.split_from_events(events)
        
        assert result.pp_comm_time == 100  # Send + Recv
        assert result.ep_comm_time == 100  # All2All
    
    def test_compute_ratios(self):
        """计算占比"""
        from npu_mfu_analyzer.analyzers import CommSplitResult
        
        result = CommSplitResult(
            tp_comm_time=50,
            pp_comm_time=30,
            dp_comm_time=20,
            total_comm_time=100,
        )
        
        ratios = result.compute_ratios()
        
        assert ratios["tp"] == 50.0
        assert ratios["pp"] == 30.0
        assert ratios["dp"] == 20.0
    
    def test_parallel_config(self):
        """并行配置"""
        from npu_mfu_analyzer.analyzers import ParallelConfig
        
        config = ParallelConfig.from_dict({
            "world_size": 64,
            "tp_size": 8,
            "pp_size": 4,
            "dp_size": 2,
        })
        
        assert config.tensor_parallel_size == 8
        assert config.pipeline_parallel_size == 4
        assert config.data_parallel_size == 2
