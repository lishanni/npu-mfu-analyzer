"""
数据加载模块测试
"""

import pytest
import tempfile
import json
import os
from pathlib import Path


class TestStreamParser:
    """StreamParser 测试"""
    
    def test_init_with_valid_path(self, tmp_path):
        """测试有效路径初始化"""
        from src.data_loader.stream_parser import StreamParser
        
        # 创建测试 JSON 文件
        test_file = tmp_path / "test.json"
        test_file.write_text('[{"name": "test", "ts": 100, "dur": 10}]')
        
        parser = StreamParser(str(test_file))
        assert parser.file_path == str(test_file)
    
    def test_iter_events_streaming(self, tmp_path):
        """测试流式解析"""
        from src.data_loader.stream_parser import StreamParser
        
        # 创建测试数据
        events = [
            {"name": "op1", "ts": 100, "dur": 10, "cat": "Kernel"},
            {"name": "op2", "ts": 200, "dur": 20, "cat": "Communication"},
        ]
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(events))
        
        parser = StreamParser(str(test_file), disable_streaming=True)
        
        result = list(parser.iter_events(show_progress=False))
        assert len(result) == 2
        assert result[0]["name"] == "op1"
        assert result[1]["name"] == "op2"
    
    def test_parse_with_filter(self, tmp_path):
        """测试带过滤的解析"""
        from src.data_loader.stream_parser import StreamParser
        
        events = [
            {"name": "op1", "ts": 100, "dur": 10, "cat": "Kernel"},
            {"name": "op2", "ts": 200, "dur": 20, "cat": "Communication"},
            {"name": "op3", "ts": 300, "dur": 30, "cat": "Kernel"},
        ]
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(events))
        
        parser = StreamParser(str(test_file), disable_streaming=True)
        
        # 只过滤 Kernel 类型
        result = parser.parse_with_filter(
            filter_func=lambda e: e.get("cat") == "Kernel",
            show_progress=False
        )
        
        assert len(result) == 2
        assert all(e["cat"] == "Kernel" for e in result)


class TestTimelineSummarizer:
    """TimelineSummarizer 测试"""
    
    def test_process_event(self):
        """测试事件处理"""
        from src.data_loader.stream_parser import TimelineSummarizer
        
        summarizer = TimelineSummarizer(max_top_events=5)
        
        events = [
            {"name": "op1", "ts": 100, "dur": 10, "cat": "Kernel"},
            {"name": "op2", "ts": 200, "dur": 50, "cat": "Communication"},
            {"name": "op3", "ts": 300, "dur": 30, "cat": "Kernel"},
        ]
        
        for event in events:
            summarizer.process_event(event)
        
        summary = summarizer.get_summary()
        
        assert summary["total_events"] == 3
        assert summary["total_duration_us"] == 90
        assert "Kernel" in summary["by_category"]
        assert summary["by_category"]["Kernel"]["count"] == 2


class TestProfilingLoader:
    """ProfilingLoader 测试"""
    
    def test_detect_empty_dir(self, tmp_path):
        """测试空目录检测"""
        from src.data_loader.profiling_loader import ProfilingLoader
        
        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()
        
        assert info.data_type == "unknown"
        assert info.rank_count == 1
    
    def test_detect_json_data(self, tmp_path):
        """测试 JSON 数据检测"""
        from src.data_loader.profiling_loader import ProfilingLoader
        
        # 创建 trace_view.json
        trace_file = tmp_path / "trace_view.json"
        trace_file.write_text('[{"name": "test"}]')
        
        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()
        
        assert info.data_type == "json"
        assert info.has_timeline


class TestDataSummarizer:
    """DataSummarizer 测试"""
    
    def test_summarize_empty_data(self, tmp_path):
        """测试空数据摘要"""
        from src.data_loader.profiling_loader import ProfilingLoader
        from src.data_loader.data_summarizer import DataSummarizer
        
        loader = ProfilingLoader(str(tmp_path))
        summarizer = DataSummarizer(loader)
        
        summary = summarizer.summarize()
        
        assert summary.data_path == str(tmp_path)
        assert summary.step_count == 0
    
    def test_to_prompt_text(self):
        """测试 Prompt 文本生成"""
        from src.data_loader.data_summarizer import ProfilingSummary, OverlapMetrics
        
        summary = ProfilingSummary(
            data_path="/test/path",
            data_type="db",
            framework="pytorch",
            rank_count=8,
            step_count=100,
            avg_step_time=50000,  # 50ms
            avg_compute_time=30000,  # 30ms
            avg_comm_time=15000,  # 15ms
            avg_free_time=5000,  # 5ms
            overlap_metrics=OverlapMetrics(overlap_ratio=60.0),
        )
        
        text = summary.to_prompt_text()
        
        assert "50.00 ms" in text
        assert "30.00 ms" in text
        assert "60.0%" in text
