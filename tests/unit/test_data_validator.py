"""
测试数据验证和错误处理功能
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.data_loader.data_validator import (
    ProfilingDataValidator,
    ProfilingDataSanitizer,
    RobustTimelineParser,
    DataQualityLevel,
    IssueType,
    validate_and_parse,
)


class TestDataValidation:
    """测试数据验证"""

    def test_validate_valid_event(self):
        """测试验证有效事件"""
        event = {
            "name": "test_event",
            "ph": "X",
            "ts": 1000,
            "dur": 500,
            "pid": "test_pid",
        }

        issues = ProfilingDataValidator.validate_timeline_event(event, 0, "test.json")
        assert len(issues) == 0

    def test_validate_missing_required_field(self):
        """测试缺少必需字段"""
        event = {
            "name": "test_event",
            "ph": "X",
            # 缺少 "ts"
            "pid": "test_pid",
        }

        issues = ProfilingDataValidator.validate_timeline_event(event, 0, "test.json")
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_FIELD
        assert issues[0].field == "ts"

    def test_validate_invalid_event_type(self):
        """测试无效的事件类型"""
        event = {
            "name": "test_event",
            "ph": "INVALID",
            "ts": 1000,
            "pid": "test_pid",
        }

        issues = ProfilingDataValidator.validate_timeline_event(event, 0, "test.json")
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.INVALID_VALUE
        assert issues[0].field == "ph"

    def test_validate_negative_duration(self):
        """测试负的持续时间"""
        event = {
            "name": "test_event",
            "ph": "X",
            "ts": 1000,
            "dur": -100,  # 负值
            "pid": "test_pid",
        }

        issues = ProfilingDataValidator.validate_timeline_event(event, 0, "test.json")
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.INVALID_VALUE
        assert issues[0].field == "dur"

    def test_validate_extreme_timestamp(self):
        """测试超出范围的时间戳"""
        event = {
            "name": "test_event",
            "ph": "X",
            "ts": 1e20,  # 超大值
            "pid": "test_pid",
        }

        issues = ProfilingDataValidator.validate_timeline_event(event, 0, "test.json")
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.INVALID_VALUE
        assert issues[0].field == "ts"


class TestDataSanitization:
    """测试数据清洗"""

    def test_sanitize_valid_event(self):
        """测试清洗有效事件"""
        from decimal import Decimal

        event = {
            "name": "test_event",
            "ph": "X",
            "ts": Decimal("1000"),
            "dur": Decimal("500"),
            "pid": "test_pid",
        }

        cleaned, valid = ProfilingDataSanitizer.sanitize_timeline_event(event, 0)
        assert valid is True
        assert isinstance(cleaned["ts"], float)
        assert isinstance(cleaned["dur"], float)
        assert cleaned["ts"] == 1000.0
        assert cleaned["dur"] == 500.0

    def test_sanitize_empty_duration(self):
        """测试空字符串持续时间"""
        event = {
            "name": "test_event",
            "ph": "X",
            "ts": "1000",
            "dur": "",  # 空字符串
            "pid": "test_pid",
        }

        cleaned, valid = ProfilingDataSanitizer.sanitize_timeline_event(event, 0)
        assert valid is True
        assert cleaned["dur"] == 0

    def test_sanitize_missing_ph(self):
        """测试缺少 ph 字段"""
        event = {
            "name": "test_event",
            "ts": "1000",
            "pid": "test_pid",
        }

        cleaned, valid = ProfilingDataSanitizer.sanitize_timeline_event(event, 0)
        assert valid is False

    def test_sanitize_metadata_event(self):
        """测试元数据事件"""
        event = {
            "name": "process_name",
            "ph": "M",
            "ts": "1000",
            "pid": "test_pid",
        }

        cleaned, valid = ProfilingDataSanitizer.sanitize_timeline_event(event, 0)
        # 元数据事件是有效的（虽然可能被某些分析过滤）
        assert valid is True
        assert cleaned["ph"] == "M"


class TestRobustParsing:
    """测试健壮解析"""

    def test_parse_valid_file(self, tmp_path):
        """测试解析有效文件"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "event1", "ph": "X", "ts": 1000, "dur": 500, "pid": "0"},
            {"name": "event2", "ph": "B", "ts": 2000, "dur": 300, "pid": "0"},
            {"name": "event3", "ph": "E", "ts": 3000, "dur": 200, "pid": "0"},
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        assert len(parsed_events) == 3
        assert report.valid_events == 3
        assert report.total_events == 3
        assert report.skipped_events == 0
        assert report.error_count == 0

    def test_parse_file_with_invalid_events(self, tmp_path):
        """测试解析包含无效事件的文件"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "valid_event", "ph": "X", "ts": 1000, "dur": 500, "pid": "0"},
            {"name": "invalid_event", "ph": "X", "ts": 1000, "dur": -100, "pid": "0"},  # 负 dur
            {"name": "another_valid", "ph": "B", "ts": 2000, "dur": 300, "pid": "0"},
            {"name": "missing_ts", "ph": "X", "dur": 200, "pid": "0"},  # 缺少 ts
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        # 应该跳过无效事件
        assert report.valid_events == 2
        assert report.total_events == 4
        assert report.skipped_events == 2
        assert report.error_count == 2  # 缺少 ts 是 error

    def test_parse_corrupted_json(self, tmp_path):
        """测试解析损坏的 JSON"""
        trace_file = tmp_path / "trace_view.json"

        # 写入不完整的 JSON
        trace_file.write_text('{"traceEvents": [{"name": "event1", "ph": "X"')

        parsed_events, report = validate_and_parse(str(trace_file))

        assert report.quality_level == DataQualityLevel.CRITICAL
        assert len(parsed_events) == 0

    def test_parse_file_with_decimal_values(self, tmp_path):
        """测试解析包含 Decimal 值的文件（ijson 问题）"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "event1", "ph": "X", "ts": "1000", "dur": "500", "pid": "0"},
            {"name": "event2", "ph": "B", "ts": "2000.5", "dur": "300.7", "pid": "0"},  # 浮点数
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        assert len(parsed_events) == 2
        assert report.valid_events == 2
        # 验证数据类型正确
        assert isinstance(parsed_events[0]["ts"], (int, float))
        assert isinstance(parsed_events[1]["ts"], (int, float))

    def test_quality_report_summary(self, tmp_path):
        """测试质量报告生成"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "valid", "ph": "X", "ts": 1000, "dur": 500, "pid": "0"},
            {"name": "negative_dur", "ph": "X", "ts": 1000, "dur": -100, "pid": "0"},
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        summary = report.to_summary()

        assert "质量等级" in summary
        # 负值 dur 现在是错误，不是警告
        assert "error: 1" in summary or "错误: 1" in summary

    def test_strict_mode(self, tmp_path):
        """测试严格模式"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "valid_event", "ph": "X", "ts": 1000, "dur": 500, "pid": "0"},
            {"name": "invalid_event", "ph": "X", "ts": 1000, "dur": -100, "pid": "0"},
        ]
        trace_file.write_text(json.dumps(events))

        parser = RobustTimelineParser(str(trace_file), strict=True)
        parsed_events, report = parser.parse()

        # 严格模式下遇到错误应该停止
        assert len(parsed_events) == 1  # 只有第一个有效事件
        assert parser.quality_report.error_count > 0

    def test_lenient_mode(self, tmp_path):
        """测试宽松模式（默认）"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "valid1", "ph": "X", "ts": 1000, "dur": 500, "pid": "0"},
            {"name": "invalid", "ph": "X", "ts": 1000, "dur": -100, "pid": "0"},
            {"name": "valid2", "ph": "B", "ts": 2000, "dur": 300, "pid": "0"},
        ]
        trace_file.write_text(json.dumps(events))

        parser = RobustTimelineParser(str(trace_file), strict=False)
        parsed_events, report = parser.parse()

        # 宽松模式应该跳过无效事件，继续处理
        assert len(parsed_events) >= 1  # 至少有有效事件
        assert report.skipped_events >= 0

    def test_data_integrity_flags(self, tmp_path):
        """测试数据完整性标志"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "kernel_op", "ph": "X", "ts": 1000, "dur": 500, "cat": "Kernel", "pid": "0"},
            {"name": "allreduce", "ph": "X", "ts": 2000, "dur": 300, "pid": "0"},
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        assert report.has_timeline_data is True
        assert report.has_kernel_data is True
        assert report.has_communication_data is True


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_file(self, tmp_path):
        """测试空文件"""
        trace_file = tmp_path / "trace_view.json"
        trace_file.write_text("[]")

        parsed_events, report = validate_and_parse(str(trace_file))

        assert len(parsed_events) == 0
        assert report.valid_events == 0
        assert report.total_events == 0

    def test_file_with_only_metadata(self, tmp_path):
        """测试只有元数据的文件"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            {"name": "process_name", "ph": "M", "pid": "1234"},
            {"name": "thread_name", "ph": "M", "tid": "5678"},
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        # 元数据事件应该被清洗为有效
        assert len(parsed_events) == 2
        assert report.valid_events == 2

    def test_mixed_valid_invalid(self, tmp_path):
        """测试混合有效和无效事件"""
        trace_file = tmp_path / "trace_view.json"

        events = [
            # 有效事件
            {"name": "valid1", "ph": "X", "ts": 1000, "dur": 500, "pid": "0"},
            # 缺少 ts
            {"name": "no_ts", "ph": "X", "dur": 200, "pid": "0"},
            # 有效
            {"name": "valid2", "ph": "B", "ts": 2000, "dur": 300, "pid": "0"},
            # 负 dur
            {"name": "neg_dur", "ph": "X", "ts": 3000, "dur": -100, "pid": "0"},
            # 空 ts
            {"name": "empty_ts", "ph": "X", "ts": "", "dur": 200, "pid": "0"},
            # 有效
            {"name": "valid3", "ph": "E", "ts": 4000, "dur": 100, "pid": "0"},
        ]
        trace_file.write_text(json.dumps(events))

        parsed_events, report = validate_and_parse(str(trace_file))

        # 应该有 3 个有效事件
        assert len(parsed_events) == 3
        assert report.valid_events == 3
        assert report.total_events == 6  # 6 个事件总数
        assert report.skipped_events == 3  # 3 个无效事件被跳过
