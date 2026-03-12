"""
测试 Decimal 类型转换修复

验证 TimelineSummarizer 能正确处理 ijson 返回的 Decimal 类型。
"""

import pytest
from decimal import Decimal
from npu_mfu_analyzer.data_loader.stream_parser import TimelineSummarizer


class TestDecimalTypeConversion:
    """测试 Decimal 类型转换"""

    def test_process_event_with_decimal_dur(self):
        """测试处理 Decimal 类型的 dur 值"""
        summarizer = TimelineSummarizer()

        # 模拟 ijson 返回的 Decimal 类型
        event = {
            "name": "test_event",
            "cat": "test_category",
            "pid": "test_pid",
            "dur": Decimal("123.456"),  # ijson 返回的 Decimal 类型
            "ts": 1000,
        }

        # 不应该抛出 TypeError
        summarizer.process_event(event)

        summary = summarizer.get_summary()

        assert summary["total_events"] == 1
        assert summary["total_duration_us"] == 123.456
        assert summary["by_category"]["test_category"]["count"] == 1
        assert summary["by_category"]["test_category"]["duration"] == 123.456

    def test_process_event_with_string_dur(self):
        """测试处理字符串类型的 dur 值"""
        summarizer = TimelineSummarizer()

        event = {
            "name": "test_event",
            "cat": "test_category",
            "pid": "test_pid",
            "dur": "789.012",  # 字符串类型
            "ts": 1000,
        }

        summarizer.process_event(event)

        summary = summarizer.get_summary()

        assert summary["total_events"] == 1
        assert summary["total_duration_us"] == 789.012

    def test_process_event_with_int_dur(self):
        """测试处理整数类型的 dur 值"""
        summarizer = TimelineSummarizer()

        event = {
            "name": "test_event",
            "cat": "test_category",
            "pid": "test_pid",
            "dur": 500,  # 整数类型
            "ts": 1000,
        }

        summarizer.process_event(event)

        summary = summarizer.get_summary()

        assert summary["total_events"] == 1
        assert summary["total_duration_us"] == 500

    def test_process_event_with_float_dur(self):
        """测试处理浮点数类型的 dur 值"""
        summarizer = TimelineSummarizer()

        event = {
            "name": "test_event",
            "cat": "test_category",
            "pid": "test_pid",
            "dur": 345.678,  # 浮点数类型
            "ts": 1000,
        }

        summarizer.process_event(event)

        summary = summarizer.get_summary()

        assert summary["total_events"] == 1
        assert summary["total_duration_us"] == 345.678

    def test_process_event_with_invalid_dur(self):
        """测试处理无效的 dur 值"""
        summarizer = TimelineSummarizer()

        event = {
            "name": "test_event",
            "cat": "test_category",
            "pid": "test_pid",
            "dur": "invalid",  # 无效字符串
            "ts": 1000,
        }

        summarizer.process_event(event)

        summary = summarizer.get_summary()

        assert summary["total_events"] == 1
        assert summary["total_duration_us"] == 0.0

    def test_extract_time_info_with_decimal(self):
        """测试 _extract_time_info 处理 Decimal 类型"""
        from npu_mfu_analyzer.data_loader.stream_parser import _extract_time_info

        event = {
            "name": "test_event",
            "cat": "test_category",
            "pid": "test_pid",
            "tid": "test_tid",
            "ts": Decimal("1000.123"),
            "dur": Decimal("500.456"),
        }

        result = _extract_time_info(event)

        assert result["ts"] == 1000.123
        assert result["dur"] == 500.456
        assert result["name"] == "test_event"
