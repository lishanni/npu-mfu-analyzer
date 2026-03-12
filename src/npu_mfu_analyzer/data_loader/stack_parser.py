"""
堆栈解析器

从 trace_view.json 的事件中提取和解析 Python/C++ 堆栈信息。
支持多种堆栈字段格式。
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter

from .stack_types import (
    StackFrame,
    HostStack,
    OperatorWithStack,
    STACK_PATTERNS,
)

logger = logging.getLogger(__name__)


class StackParser:
    """
    堆栈解析器

    从 trace_view.json 事件中提取堆栈信息并识别特征模式。

    Usage:
        parser = StackParser()
        stack = parser.parse_from_event(event)
        if stack:
            print(stack.get_source_type())  # "torch_compile", "eager", etc.
    """

    # 堆栈字段可能的名称（按优先级）
    PYTHON_STACK_FIELDS = [
        "python_stack",
        "Python Stack",
        "call_stack",
        "Call stack",
        "stack",
        "backtrace",
    ]

    CPP_STACK_FIELDS = [
        "c++_stack",
        "C++ Stack",
        "cpp_stack",
        "C++_stack",
        "cpp_backtrace",
    ]

    def __init__(self, max_depth: int = 50):
        """
        Args:
            max_depth: 最大解析堆栈深度（防止内存溢出）
        """
        self.max_depth = max_depth
        self._pattern_cache: Dict[str, str] = {}  # 缓存函数名到来源类型的映射

    def parse_from_event(self, event: Dict[str, Any]) -> Optional[HostStack]:
        """
        从事件中解析堆栈信息

        Args:
            event: trace_view.json 中的单个事件

        Returns:
            HostStack 对象，如果无堆栈信息则返回 None
        """
        args = event.get("args", {})
        if not args or not isinstance(args, dict):
            return None

        # 提取 Python 堆栈
        python_stack_str = self._extract_stack_field(args, self.PYTHON_STACK_FIELDS)
        python_frames = self._parse_stack_string(python_stack_str) if python_stack_str else []

        # 提取 C++ 堆栈
        cpp_stack_str = self._extract_stack_field(args, self.CPP_STACK_FIELDS)
        cpp_frames = self._parse_stack_string(cpp_stack_str) if cpp_stack_str else []

        if not python_frames and not cpp_frames:
            return None

        # 创建 HostStack 对象
        host_stack = HostStack(
            python_stack=python_frames[:self.max_depth],
            cpp_stack=cpp_frames[:self.max_depth],
            raw_python_stack=python_stack_str[:2000] if python_stack_str else None,
            raw_cpp_stack=cpp_stack_str[:2000] if cpp_stack_str else None,
        )

        # 识别特征
        self._identify_features(host_stack)

        return host_stack

    def _extract_stack_field(self, args: Dict[str, Any], field_names: List[str]) -> Optional[str]:
        """从 args 中提取指定字段的堆栈"""
        for name in field_names:
            if name in args:
                value = args[name]
                if isinstance(value, str) and value.strip():
                    return value
                elif isinstance(value, list):
                    # 某些格式可能是列表
                    return "\n".join(str(item) for item in value)
        return None

    def _parse_stack_string(self, stack_str: str) -> List[StackFrame]:
        """
        解析堆栈字符串为 StackFrame 列表

        支持多种格式：
        - "File \"path/to/file.py\", line 123, in function_name"
        - "function_name (path/to/file.py:123)"
        - "path/to/file.py:123 in function_name"
        """
        frames = []

        # 常见堆栈格式正则
        patterns = [
            # Python 标准格式: File "xxx", line N, in func
            r'File\s+"([^"]+)",\s*line\s+(\d+),\s*in\s+(\S+)',
            # 简化格式: func (file:line)
            r'(\S+)\s*\(([^:)]+):(\d+)\)',
            # 另一种格式: file:line in func
            r'([^:]+):(\d+)\s+in\s+(\S+)',
            # GDB 格式: #N func () at file:line
            r'#\d+\s+(\S+).*?\(([^:)]+):(\d+)\)',
        ]

        lines = stack_str.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    if len(groups) == 3:
                        # 根据捕获组顺序调整
                        if pattern.startswith('File'):
                            filename, lineno, function = groups
                        elif pattern.startswith('#'):
                            function, filename, lineno = groups
                        else:
                            # 根据第一个字符判断
                            if groups[0].endswith('.py') or groups[0].endswith('.cpp'):
                                filename, lineno, function = groups
                            else:
                                function, filename, lineno = groups

                        try:
                            lineno_int = int(lineno)
                        except ValueError:
                            lineno_int = 0

                        frames.append(StackFrame(
                            function=function,
                            filename=filename,
                            lineno=lineno_int,
                        ))
                        break

        return frames

    def _identify_features(self, host_stack: HostStack) -> None:
        """识别堆栈特征"""
        # 收集所有函数名
        all_functions = [f.function for f in host_stack.python_stack]
        all_functions.extend(f.function for f in host_stack.cpp_stack)

        # 转为字符串用于模式匹配
        stack_text = " ".join(all_functions)

        # 按优先级识别（先识别更具体的模式）
        # 1. 融合算子
        if self._match_patterns(stack_text, STACK_PATTERNS["fusion_op"]["patterns"]):
            host_stack.is_fusion_op = True
            # 尝试识别融合算子名称
            for pattern in STACK_PATTERNS["fusion_op"]["patterns"]:
                if pattern in stack_text:
                    host_stack.fusion_op_name = pattern
                    break

        # 2. torch.compile
        if self._match_patterns(stack_text, STACK_PATTERNS["torch_compile"]["patterns"]):
            host_stack.is_torch_compile = True

        # 3. mindspeed
        if self._match_patterns(stack_text, STACK_PATTERNS["mindspeed"]["patterns"]):
            host_stack.is_mindspeed = True

        # 4. torch-ascend
        if self._match_patterns(stack_text, STACK_PATTERNS["torch_ascend"]["patterns"]):
            host_stack.is_torch_ascend = True

        # 5. 分布式
        if self._match_patterns(stack_text, STACK_PATTERNS["distributed"]["patterns"]):
            host_stack.is_distributed = True

        # 6. 优化器
        if self._match_patterns(stack_text, STACK_PATTERNS["optimizer"]["patterns"]):
            host_stack.is_optimizer = True

        # 7. eager 模式（排除 torch.compile）
        if not host_stack.is_torch_compile:
            if self._match_patterns(stack_text, STACK_PATTERNS["eager"]["patterns"]):
                # 检查排除条件
                excludes = STACK_PATTERNS["eager"].get("excludes", [])
                if not any(ex in stack_text for ex in excludes):
                    host_stack.is_eager = True

    def _match_patterns(self, text: str, patterns: List[str]) -> bool:
        """检查文本是否匹配任一模式"""
        return any(pattern in text for pattern in patterns)

    def get_source_type(self, function_name: str) -> str:
        """
        根据函数名获取来源类型（使用缓存）

        Args:
            function_name: 函数名

        Returns:
            来源类型字符串
        """
        if function_name in self._pattern_cache:
            return self._pattern_cache[function_name]

        # 按优先级匹配
        for source_type, config in STACK_PATTERNS.items():
            patterns = config.get("patterns", [])
            if any(p in function_name for p in patterns):
                self._pattern_cache[function_name] = source_type
                return source_type

        result = "unknown"
        self._pattern_cache[function_name] = result
        return result


class StackPatternDiscovery:
    """
    堆栈模式自动发现

    从实际数据中自动识别高频堆栈模式，用于扩展 STACK_PATTERNS。
    """

    def __init__(self, min_frequency: int = 5):
        """
        Args:
            min_frequency: 最小频率阈值
        """
        self.min_frequency = min_frequency
        self.function_counter: Counter = Counter()
        self.pattern_groups: Dict[str, Set[str]] = {}

    def process_event(self, event: Dict[str, Any], parser: StackParser) -> None:
        """处理单个事件，收集堆栈模式"""
        stack = parser.parse_from_event(event)
        if not stack:
            return

        for frame in stack.python_stack:
            self.function_counter[frame.function] += 1

    def discover_patterns(self) -> Dict[str, List[str]]:
        """
        发现高频堆栈模式

        Returns:
            {pattern_type: [function_names]}
        """
        discovered = {
            "high_frequency": [],
            "potential_compile": [],
            "potential_fusion": [],
            "potential_mindspeed": [],
        }

        for func, count in self.function_counter.most_common(100):
            if count < self.min_frequency:
                continue

            discovered["high_frequency"].append(f"{func} ({count})")

            # 识别潜在模式
            if "compile" in func.lower() or "dynamo" in func.lower():
                discovered["potential_compile"].append(func)
            if "fuse" in func.lower() or "gmm" in func.lower():
                discovered["potential_fusion"].append(func)
            if "megatron" in func.lower() or "parallel" in func.lower():
                discovered["potential_mindspeed"].append(func)

        return discovered

    def get_suggested_patterns(self) -> Dict[str, List[str]]:
        """
        获取建议添加到 STACK_PATTERNS 的新模式

        Returns:
            {source_type: [suggested_patterns]}
        """
        discovered = self.discover_patterns()
        suggestions = {}

        # 分析潜在的新模式
        if discovered["potential_compile"]:
            suggestions["torch_compile"] = discovered["potential_compile"][:5]

        if discovered["potential_fusion"]:
            suggestions["fusion_op"] = discovered["potential_fusion"][:5]

        if discovered["potential_mindspeed"]:
            suggestions["mindspeed"] = discovered["potential_mindspeed"][:5]

        return suggestions


def extract_stack_from_events(
    events: List[Dict[str, Any]],
    max_events: int = 10000,
) -> List[OperatorWithStack]:
    """
    从事件列表中提取带堆栈信息的算子

    Args:
        events: 事件列表
        max_events: 最大处理事件数

    Returns:
        OperatorWithStack 列表
    """
    parser = StackParser()
    operators = []

    for i, event in enumerate(events[:max_events]):
        # 只处理有名称和耗时的事件
        name = event.get("name", "")
        dur = event.get("dur", 0)
        if not name or not dur:
            continue

        # 解析堆栈
        stack = parser.parse_from_event(event)

        if stack:
            op = OperatorWithStack(
                name=name,
                dur=float(dur) if dur else 0.0,
                cat=event.get("cat", ""),
                ts=float(event.get("ts", 0)),
                pid=event.get("pid", ""),
                tid=event.get("tid", ""),
                host_stack=stack,
                raw_event=event,
            )
            operators.append(op)

    logger.info(f"Extracted {len(operators)} operators with stack info from {min(len(events), max_events)} events")
    return operators


def analyze_stack_distribution(
    operators: List[OperatorWithStack],
) -> Dict[str, Any]:
    """
    分析堆栈来源分布

    Args:
        operators: 带堆栈信息的算子列表

    Returns:
        分布统计字典
    """
    distribution = Counter()
    source_operators: Dict[str, List[str]] = {}

    for op in operators:
        if op.host_stack:
            source_type = op.host_stack.get_source_type()
            distribution[source_type] += 1

            if source_type not in source_operators:
                source_operators[source_type] = []
            source_operators[source_type].append(op.name)

    # 去重并排序
    for source_type in source_operators:
        counter = Counter(source_operators[source_type])
        source_operators[source_type] = [f"{name} ({count})" for name, count in counter.most_common(20)]

    return {
        "distribution": dict(distribution),
        "total_operators": len(operators),
        "operators_by_source": source_operators,
    }
