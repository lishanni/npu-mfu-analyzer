"""
数据验证和错误处理模块

提供 Profiling 数据的验证、容错处理和质量报告功能。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 完整且格式正确
    GOOD = "good"           # 轻微问题，不影响分析
    FAIR = "fair"           # 中等问题，部分分析受限
    POOR = "poor"           # 严重问题，分析受限
    CRITICAL = "critical"   # 数据无法使用


class IssueType(Enum):
    """问题类型"""
    MISSING_FIELD = "missing_field"           # 缺少必要字段
    INVALID_TYPE = "invalid_type"             # 类型错误
    INVALID_VALUE = "invalid_value"           # 值超出范围
    CORRUPTED_DATA = "corrupted_data"         # 数据损坏
    PARSE_ERROR = "parse_error"               # 解析错误
    INCONSISTENT = "inconsistent"             # 数据不一致


@dataclass
class DataIssue:
    """数据问题记录"""
    issue_type: IssueType
    severity: str  # "error", "warning", "info"
    location: str  # 问题位置（文件路径、行号等）
    field: str = ""
    message: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "severity": self.severity,
            "location": self.location,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class DataQualityReport:
    """数据质量报告"""
    file_path: str = ""
    quality_level: DataQualityLevel = DataQualityLevel.EXCELLENT
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: List[DataIssue] = field(default_factory=list)

    # 统计信息
    total_events: int = 0
    valid_events: int = 0
    skipped_events: int = 0

    # 数据完整性
    has_timeline_data: bool = False
    has_memory_data: bool = False
    has_communication_data: bool = False
    has_kernel_data: bool = False

    def add_issue(self, issue: DataIssue):
        """添加问题"""
        self.issues.append(issue)
        self.total_issues += 1

        if issue.severity == "error":
            self.error_count += 1
        elif issue.severity == "warning":
            self.warning_count += 1
        elif issue.severity == "info":
            self.info_count += 1

        # 更新质量等级
        self._update_quality_level()

    def _update_quality_level(self):
        """根据问题数量更新质量等级"""
        if self.error_count > 0:
            if self.error_count > 100:
                self.quality_level = DataQualityLevel.CRITICAL
            elif self.error_count > 20:
                self.quality_level = DataQualityLevel.POOR
            else:
                self.quality_level = DataQualityLevel.FAIR
        elif self.warning_count > 50:
            self.quality_level = DataQualityLevel.FAIR
        elif self.warning_count > 10:
            self.quality_level = DataQualityLevel.GOOD
        else:
            self.quality_level = DataQualityLevel.EXCELLENT

    def to_summary(self) -> str:
        """生成质量报告摘要"""
        lines = [
            f"# 数据质量报告",
            f"",
            f"**文件**: {self.file_path}",
            f"**质量等级**: {self.quality_level.value.upper()}",
            f"",
            f"## 问题统计",
            f"- 总问题数: {self.total_issues}",
            f"- 错误: {self.error_count}",
            f"- 警告: {self.warning_count}",
            f"- 信息: {self.info_count}",
            f"",
        ]

        if self.valid_events < self.total_events:
            lines.append(f"## 数据完整性")
            lines.append(f"- 总事件数: {self.total_events}")
            lines.append(f"- 有效事件: {self.valid_events}")
            lines.append(f"- 跳过事件: {self.skipped_events}")
            lines.append(f"- 有效率: {self.valid_events / self.total_events * 100 if self.total_events > 0 else 0:.1f}%")
            lines.append(f"")

        if self.issues:
            lines.append(f"## 主要问题")
            # 按严重程度和类型分组
            error_issues = [i for i in self.issues if i.severity == "error"][:5]
            warning_issues = [i for i in self.issues if i.severity == "warning"][:5]

            if error_issues:
                lines.append(f"### 错误")
                for issue in error_issues:
                    lines.append(f"- **{issue.issue_type.value}**: {issue.message}")
                    if issue.suggestion:
                        lines.append(f"  建议: {issue.suggestion}")

            if warning_issues:
                lines.append(f"### 警告")
                for issue in warning_issues:
                    lines.append(f"- **{issue.issue_type.value}**: {issue.message}")

        return "\n".join(lines)


class ProfilingDataValidator:
    """
    Profiling 数据验证器

    提供：
    1. 数据格式验证
    2. 数据完整性检查
    3. 数据范围验证
    4. 数据一致性检查
    """

    # trace_view.json 必需字段
    TIMELINE_REQUIRED_FIELDS = {"name", "ph", "ts", "pid"}

    # trace_view.json 可选字段
    TIMELINE_OPTIONAL_FIELDS = {"dur", "tid", "cat", "args"}

    # 事件类型（ph）
    VALID_EVENT_TYPES = {
        "X", "B", "E",  # Duration events
        "I", "C", "F",  # Instant events
        "M",           # Metadata
    }

    # 合理的值范围
    REASONABLE_TIMESTAMP_RANGE = (0, 1e15)  # 0 ~ 1e15 微秒
    REASONABLE_DURATION_RANGE = (0, 1e12)   # 0 ~ 1e12 微秒

    @classmethod
    def validate_timeline_event(cls, event: Dict[str, Any], index: int, file_path: str) -> List[DataIssue]:
        """
        验证单个 timeline 事件

        Args:
            event: 事件字典
            index: 事件索引
            file_path: 文件路径

        Returns:
            问题列表
        """
        issues = []
        location = f"{file_path}:event#{index}"

        # 检查必需字段
        for field in cls.TIMELINE_REQUIRED_FIELDS:
            if field not in event:
                issues.append(DataIssue(
                    issue_type=IssueType.MISSING_FIELD,
                    severity="error",
                    location=location,
                    field=field,
                    message=f"缺少必需字段 '{field}'",
                    suggestion=f"确保事件包含所有必需字段: {cls.TIMELINE_REQUIRED_FIELDS}",
                ))

        # 检查事件类型
        if "ph" in event:
            ph = event["ph"]
            if ph not in cls.VALID_EVENT_TYPES:
                issues.append(DataIssue(
                    issue_type=IssueType.INVALID_VALUE,
                    severity="warning",
                    location=location,
                    field="ph",
                    message=f"未知的事件类型 '{ph}'",
                    suggestion=f"有效类型: {cls.VALID_EVENT_TYPES}",
                ))

        # 检查时间戳
        if "ts" in event:
            ts = event["ts"]
            try:
                ts_float = float(ts)
                if not (cls.REASONABLE_TIMESTAMP_RANGE[0] <= ts_float <= cls.REASONABLE_TIMESTAMP_RANGE[1]):
                    issues.append(DataIssue(
                        issue_type=IssueType.INVALID_VALUE,
                        severity="warning",
                        location=location,
                        field="ts",
                        message=f"时间戳超出合理范围: {ts}",
                        suggestion="检查时间戳单位是否正确（应为微秒）",
                    ))
            except (TypeError, ValueError):
                issues.append(DataIssue(
                    issue_type=IssueType.INVALID_TYPE,
                    severity="error",
                    location=location,
                    field="ts",
                    message=f"时间戳类型错误: {type(ts).__name__}",
                    suggestion="时间戳应为数字类型",
                ))

        # 检查持续时间
        if "dur" in event:
            dur = event["dur"]
            try:
                dur_float = float(dur)
                if dur_float < 0:
                    issues.append(DataIssue(
                        issue_type=IssueType.INVALID_VALUE,
                        severity="error",
                        location=location,
                        field="dur",
                        message=f"持续时间不能为负: {dur}",
                        suggestion="持续时间应为非负数",
                    ))
                elif dur_float > cls.REASONABLE_DURATION_RANGE[1]:
                    issues.append(DataIssue(
                        issue_type=IssueType.INVALID_VALUE,
                        severity="warning",
                        location=location,
                        field="dur",
                        message=f"持续时间异常大: {dur}",
                        suggestion="检查单位是否正确（应为微秒）",
                    ))
            except (TypeError, ValueError):
                issues.append(DataIssue(
                    issue_type=IssueType.INVALID_TYPE,
                    severity="warning",
                    location=location,
                    field="dur",
                    message=f"持续时间类型错误: {type(dur).__name__}",
                    suggestion="持续时间应为数字类型",
                ))

        return issues


class ProfilingDataSanitizer:
    """
    Profiling 数据清洗器

    提供：
    1. 容错解析
    2. 数据修复
    3. 部分数据降级分析
    """

    @staticmethod
    def sanitize_timeline_event(event: Dict[str, Any], index: int) -> Tuple[Dict[str, Any], bool]:
        """
        清洗单个 timeline 事件

        Args:
            event: 原始事件字典
            index: 事件索引

        Returns:
            (清洗后的事件, 是否有效)
        """
        # 复制事件，避免修改原始数据
        cleaned = dict(event)

        # Accept compact trace-like events used by lightweight fixtures and tests.
        # Chrome trace complete events use ph="X"; pid can safely default to 0 when
        # the source only contains name/cat/ts/dur.
        if "ph" not in cleaned and "ts" in cleaned and ("dur" in cleaned or "cat" in cleaned):
            cleaned["ph"] = "X"
        if "pid" not in cleaned and cleaned.get("ph") != "M":
            cleaned["pid"] = 0

        # 首先检查必需字段
        # 元数据事件 (ph='M') 不需要 'ts' 字段，且可以使用 tid 代替 pid
        required_fields = ["name", "ph"]
        for field in required_fields:
            if field not in cleaned:
                return cleaned, False

        # 对于非元数据事件，pid 和 ts 是必需的
        # 对于元数据事件，pid 或 tid 至少需要一个
        if cleaned.get("ph") == "M":
            if "pid" not in cleaned and "tid" not in cleaned:
                return cleaned, False
        else:
            if "pid" not in cleaned:
                return cleaned, False
            if "ts" not in cleaned:
                return cleaned, False

        # 修复常见问题
        # 1. 处理 Decimal 类型
        from decimal import Decimal
        if "dur" in cleaned and isinstance(cleaned["dur"], Decimal):
            cleaned["dur"] = float(cleaned["dur"])
        if "ts" in cleaned and isinstance(cleaned["ts"], Decimal):
            cleaned["ts"] = float(cleaned["ts"])

        # 2. 处理字符串类型的数字（常见于某些 Profiling 工具）
        if "ts" in cleaned and isinstance(cleaned["ts"], str) and cleaned["ts"]:
            try:
                cleaned["ts"] = float(cleaned["ts"])
            except ValueError:
                pass  # 保持原样，后续验证会处理
        if "dur" in cleaned and isinstance(cleaned["dur"], str) and cleaned["dur"]:
            try:
                cleaned["dur"] = float(cleaned["dur"])
            except ValueError:
                pass  # 保持原样，后续验证会处理

        # 3. 处理空字符串的 dur/ts
        if "dur" in cleaned and cleaned["dur"] == "":
            cleaned["dur"] = 0
        if "ts" in cleaned and cleaned["ts"] == "":
            return cleaned, False

        # 3. 验证基本有效性
        try:
            # 元数据事件 (ph='M') 不需要验证 ts/dur
            if cleaned.get("ph") != "M":
                # 尝试转换关键字段
                ts_val = float(cleaned["ts"])
                if "dur" in cleaned and cleaned["dur"]:
                    dur_val = float(cleaned["dur"])
                    # 拒绝负值持续时间
                    if dur_val < 0:
                        return cleaned, False
        except (TypeError, ValueError):
            return cleaned, False

        return cleaned, True


class RobustTimelineParser:
    """
    健壮的 Timeline 解析器

    提供：
    1. 损坏数据的容错解析
    2. 详细的错误报告
    3. 部分数据降级分析
    """

    def __init__(self, file_path: str, strict: bool = False):
        """
        Args:
            file_path: trace_view.json 文件路径
            strict: 严格模式（遇到错误立即停止）
        """
        self.file_path = file_path
        self.strict = strict
        self.quality_report = DataQualityReport(file_path=file_path)

    def parse(self) -> Tuple[List[Dict[str, Any]], DataQualityReport]:
        """
        解析 trace_view.json 文件

        Returns:
            (事件列表, 质量报告)
        """
        events = []
        self.quality_report = DataQualityReport(file_path=self.file_path)

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # 尝试解析 JSON
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    self.quality_report.add_issue(DataIssue(
                        issue_type=IssueType.PARSE_ERROR,
                        severity="error",
                        location=self.file_path,
                        message=f"JSON 解析失败: {e}",
                        suggestion="检查文件是否损坏或格式是否正确",
                    ))
                    # JSON 解析失败直接设置为 CRITICAL
                    self.quality_report.quality_level = DataQualityLevel.CRITICAL
                    if self.strict:
                        raise
                    return [], self.quality_report

                # 检查数据结构
                if isinstance(data, dict):
                    # 可能是 {"traceEvents": [...]} 格式
                    if "traceEvents" in data:
                        raw_events = data["traceEvents"]
                    else:
                        self.quality_report.add_issue(DataIssue(
                            issue_type=IssueType.INVALID_TYPE,
                            severity="error",
                            location=self.file_path,
                            message="未知的数据格式",
                            suggestion="trace_view.json 应为 JSON 数组或包含 traceEvents 字段的对象",
                        ))
                        return [], self.quality_report
                elif isinstance(data, list):
                    raw_events = data
                else:
                    self.quality_report.add_issue(DataIssue(
                        issue_type=IssueType.INVALID_TYPE,
                        severity="error",
                        location=self.file_path,
                        message=f"意外的数据类型: {type(data).__name__}",
                        suggestion="trace_view.json 应为 JSON 数组",
                    ))
                    return [], self.quality_report

                self.quality_report.total_events = len(raw_events)

                # 逐个解析事件
                for i, raw_event in enumerate(raw_events):
                    # 验证事件
                    issues = ProfilingDataValidator.validate_timeline_event(
                        raw_event, i, self.file_path
                    )
                    for issue in issues:
                        self.quality_report.add_issue(issue)

                    # 严格模式下遇到错误停止
                    if self.strict and any(i.severity == "error" for i in issues):
                        logger.error(f"Strict mode: stopping at event {i} due to error")
                        break

                    # 清洗事件
                    cleaned_event, is_valid = ProfilingDataSanitizer.sanitize_timeline_event(
                        raw_event, i
                    )

                    if is_valid:
                        events.append(cleaned_event)
                        self.quality_report.valid_events += 1
                    else:
                        self.quality_report.skipped_events += 1

                # 更新数据完整性标志
                self.quality_report.has_timeline_data = len(events) > 0
                self.quality_report.has_kernel_data = any(
                    e.get("cat") == "Kernel" or e.get("ph") in ["X", "B", "E"]
                    for e in events
                )
                self.quality_report.has_communication_data = any(
                    "allreduce" in str(e.get("name", "")).lower()
                    or "alltoall" in str(e.get("name", "")).lower()
                    for e in events
                )

        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
            self.quality_report.add_issue(DataIssue(
                issue_type=IssueType.PARSE_ERROR,
                severity="error",
                location=self.file_path,
                message=f"解析异常: {e}",
                suggestion="检查文件权限和格式",
            ))

        return events, self.quality_report


def validate_and_parse(
    file_path: str,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], DataQualityReport]:
    """
    验证并解析 trace_view.json 文件

    Args:
        file_path: trace_view.json 文件路径
        strict: 严格模式（遇到错误立即停止）

    Returns:
        (事件列表, 质量报告)
    """
    parser = RobustTimelineParser(file_path, strict=strict)
    return parser.parse()
