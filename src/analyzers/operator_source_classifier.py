"""
算子来源分类器

基于堆栈特征识别算子来源类型，支持扩展的模式匹配和自动发现。
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import re

from src.data_loader.stack_types import (
    HostDeviceChain,
    CorrelationStats,
    HostStack,
    STACK_PATTERNS,
)

logger = logging.getLogger(__name__)


# 扩展版堆栈模式（包含 mindspeed、torch-ascend 等）
EXTENDED_STACK_PATTERNS = {
    # torch.compile 图模式
    "torch_compile": {
        "patterns": [
            "CompiledFunctionBackward",
            "CompiledFunction",
            "CUDAGraph",
            "torch._dynamo",
            "torch._inductor",
            "TorchDynamo",
            "AOTAutograd",
            " dynamo_wrapped",
            "_dynamo_eval_frame",
            "OptimizedModule",
            "compile_inner",
        ],
        "excludes": [],
        "label": "torch.compile 图模式",
        "color": "#FF6B6B",  # 红色
    },

    # 融合算子（昇腾原生）
    "fusion_op": {
        "patterns": [
            "NPUGroupedLinearGMM",
            "NPUGroupedMatmul",
            "FlashAttention",
            "FusedMatmul",
            "aclnnGroupedMatmul",
            "FusedScaleMaskSoftmax",
            "FusedLayerNorm",
            "FusedRMSNorm",
            "npu_scaled_masked_softmax",
            "npu_fusion",
            "npu_rotary_mul",
            "npu_scaled_masked_softmax",
            "FusedMLP",
            "FusedCrossEntropy",
        ],
        "excludes": [],
        "label": "融合算子",
        "color": "#4ECDC4",  # 青色
    },

    # eager 模式
    "eager": {
        "patterns": [
            "aten::",
            "torch.ops.",
            "torch.nn.functional",
            "torch.nn.modules",
        ],
        "excludes": [
            "CompiledFunction",
            "torch._dynamo",
            "torch._inductor",
            "dynamo_wrapped",
        ],
        "label": "Eager 模式",
        "color": "#45B7D1",  # 蓝色
    },

    # mindspeed 相关
    "mindspeed": {
        "patterns": [
            "mindspeed.",
            "mindspeed_",
            "megatron.",
            "Megatron.",
            "transformer_module",
            "ParallelMLP",
            "ParallelAttention",
            "ColumnParallelLinear",
            "RowParallelLinear",
            "VocabParallelEmbedding",
            "TensorParallel",
            "PipelineParallel",
            "SequenceParallel",
            "TransformerBlock",
            "transformer_engine",
        ],
        "excludes": [],
        "label": "MindSpeed/Megatron",
        "color": "#96CEB4",  # 绿色
    },

    # torch-ascend / CANN 相关
    "torch_ascend": {
        "patterns": [
            "torch_npu",
            "aten_npu",
            "aclnn",
            "AscendCL",
            "aclOp",
            "OpRunner",
            "AclOpExecutor",
            "cann",
            "npu::",
            "_npu_",
        ],
        "excludes": [],
        "label": "Torch-Ascend/CANN",
        "color": "#FFEAA7",  # 黄色
    },

    # 分布式通信相关
    "distributed": {
        "patterns": [
            "torch.distributed",
            "ProcessGroup",
            "HCCL",
            "hccl",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "broadcast",
            "ProcessGroupHCCL",
            "ProcessGroupNCCL",
            "DistributedDataParallel",
        ],
        "excludes": [],
        "label": "分布式通信",
        "color": "#DDA0DD",  # 紫色
    },

    # 优化器相关
    "optimizer": {
        "patterns": [
            "Optimizer.step",
            "AdamW",
            "Adam",
            "LAMB",
            "FusedAdam",
            "FusedLAMB",
            "_fused_adam",
            "_multi_tensor_adam",
            "optimizer_step",
            "backward",
        ],
        "excludes": [],
        "label": "优化器",
        "color": "#F39C12",  # 橙色
    },

    # 数据加载相关
    "dataloader": {
        "patterns": [
            "DataLoader",
            "dataloader",
            "Dataset",
            "Iterator",
            "collate",
            "prefetch",
        ],
        "excludes": [],
        "label": "数据加载",
        "color": "#BDC3C7",  # 灰色
    },
}


@dataclass
class ClassificationResult:
    """分类结果"""
    source_type: str
    confidence: float           # 置信度 0-1
    matched_patterns: List[str]
    label: str
    color: str


@dataclass
class SourceChange:
    """来源变化"""
    operator_name: str
    source_a: str              # 版本 A 的来源
    source_b: str              # 版本 B 的来源
    is_change: bool            # 是否有变化
    change_type: str           # "mode_switch", "new", "removed", "unchanged"


class OperatorSourceClassifier:
    """
    算子来源分类器

    根据堆栈特征识别算子来源类型。

    Usage:
        classifier = OperatorSourceClassifier()
        source_type = classifier.classify(chain)
        stats = classifier.build_stats(chains)
    """

    def __init__(self, custom_patterns: Optional[Dict[str, Dict]] = None):
        """
        Args:
            custom_patterns: 自定义模式，会与默认模式合并
        """
        self.patterns = EXTENDED_STACK_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # 编译正则表达式以提高性能
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for source_type, config in self.patterns.items():
            self._compiled_patterns[source_type] = [
                re.compile(re.escape(p)) for p in config.get("patterns", [])
            ]

        # 缓存分类结果
        self._cache: Dict[str, ClassificationResult] = {}

    def classify(self, chain: HostDeviceChain) -> ClassificationResult:
        """
        根据堆栈特征识别算子来源

        Args:
            chain: HostDeviceChain 对象

        Returns:
            ClassificationResult
        """
        # 构建缓存 key
        cache_key = f"{chain.torch_op_name}_{hash(tuple(chain.python_stack[:5]))}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # 收集堆栈文本
        stack_text = " ".join(chain.python_stack + chain.cpp_stack)
        stack_text_lower = stack_text.lower()

        best_match: Optional[ClassificationResult] = None
        best_score = 0

        for source_type, config in self.patterns.items():
            patterns = config.get("patterns", [])
            excludes = config.get("excludes", [])

            # 检查排除模式
            if any(ex.lower() in stack_text_lower for ex in excludes):
                continue

            # 计算匹配分数
            matched = []
            for pattern in patterns:
                if pattern.lower() in stack_text_lower:
                    matched.append(pattern)

            if matched:
                score = len(matched) / max(len(patterns), 1)

                # 优先选择更具体的匹配
                if source_type == "fusion_op" and matched:
                    score += 0.5  # 融合算子优先级更高

                if score > best_score:
                    best_score = score
                    best_match = ClassificationResult(
                        source_type=source_type,
                        confidence=min(score, 1.0),
                        matched_patterns=matched,
                        label=config.get("label", source_type),
                        color=config.get("color", "#999999"),
                    )

        # 如果没有匹配，标记为 unknown
        if best_match is None:
            best_match = ClassificationResult(
                source_type="unknown",
                confidence=0.0,
                matched_patterns=[],
                label="未知来源",
                color="#999999",
            )

        self._cache[cache_key] = best_match
        return best_match

    def classify_batch(
        self,
        chains: List[HostDeviceChain],
    ) -> List[ClassificationResult]:
        """
        批量分类

        Args:
            chains: HostDeviceChain 列表

        Returns:
            ClassificationResult 列表
        """
        return [self.classify(chain) for chain in chains]

    def build_stats(
        self,
        chains: List[HostDeviceChain],
    ) -> CorrelationStats:
        """
        构建关联统计

        Args:
            chains: HostDeviceChain 列表

        Returns:
            CorrelationStats
        """
        stats = CorrelationStats()
        stats.total_chains = len(chains)

        for chain in chains:
            result = self.classify(chain)

            # 更新 chain 的 source_type
            chain.source_type = result.source_type

            # 按来源类型统计
            stats.by_source_type[result.source_type] = stats.by_source_type.get(result.source_type, 0) + 1

            # 按算子名称统计
            op_name = chain.device_op_name or chain.torch_op_name
            stats.by_operator[op_name] = stats.by_operator.get(op_name, 0) + 1

            # 建立算子来源映射
            stats.operator_source_map[op_name] = result.source_type

            # 分类收集
            if result.source_type == "eager":
                stats.eager_ops.append(op_name)
            elif result.source_type == "torch_compile":
                stats.compile_ops.append(op_name)
            elif result.source_type == "fusion_op":
                stats.fusion_ops.append(op_name)
            elif result.source_type == "mindspeed":
                stats.mindspeed_ops.append(op_name)

        # 去重并限制数量
        stats.eager_ops = list(set(stats.eager_ops))[:20]
        stats.compile_ops = list(set(stats.compile_ops))[:20]
        stats.fusion_ops = list(set(stats.fusion_ops))[:20]
        stats.mindspeed_ops = list(set(stats.mindspeed_ops))[:20]

        return stats

    def compute_source_changes(
        self,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
    ) -> List[SourceChange]:
        """
        计算两个版本之间的来源变化

        Args:
            chains_a: 版本 A 的调用链
            chains_b: 版本 B 的调用链

        Returns:
            SourceChange 列表
        """
        # 建立算子来源映射
        source_map_a = {}
        for chain in chains_a:
            result = self.classify(chain)
            op_name = chain.device_op_name or chain.torch_op_name
            source_map_a[op_name] = result.source_type

        source_map_b = {}
        for chain in chains_b:
            result = self.classify(chain)
            op_name = chain.device_op_name or chain.torch_op_name
            source_map_b[op_name] = result.source_type

        changes = []

        # 检查所有算子
        all_ops = set(source_map_a.keys()) | set(source_map_b.keys())

        for op_name in all_ops:
            source_a = source_map_a.get(op_name, "")
            source_b = source_map_b.get(op_name, "")

            if source_a != source_b:
                if not source_a:
                    change_type = "new"
                elif not source_b:
                    change_type = "removed"
                else:
                    change_type = "mode_switch"

                changes.append(SourceChange(
                    operator_name=op_name,
                    source_a=source_a,
                    source_b=source_b,
                    is_change=True,
                    change_type=change_type,
                ))

        # 按变化类型排序
        priority = {"mode_switch": 0, "new": 1, "removed": 2}
        changes.sort(key=lambda c: (priority.get(c.change_type, 99), c.operator_name))

        return changes


class StackPatternDiscovery:
    """
    自动发现新的堆栈模式

    从实际数据中自动识别高频堆栈模式，用于扩展 EXTENDED_STACK_PATTERNS。
    """

    def __init__(self, min_frequency: int = 5):
        """
        Args:
            min_frequency: 最小频率阈值
        """
        self.min_frequency = min_frequency
        self.function_counter: Counter = Counter()
        self.pattern_candidates: Dict[str, Set[str]] = defaultdict(set)

    def process_chain(self, chain: HostDeviceChain) -> None:
        """处理单个调用链"""
        for func in chain.python_stack:
            self.function_counter[func] += 1

            # 识别潜在的模式组
            if "compile" in func.lower() or "dynamo" in func.lower():
                self.pattern_candidates["torch_compile"].add(func)
            elif "fuse" in func.lower() or "gmm" in func.lower():
                self.pattern_candidates["fusion_op"].add(func)
            elif "megatron" in func.lower() or "parallel" in func.lower():
                self.pattern_candidates["mindspeed"].add(func)
            elif "npu" in func.lower() or "acl" in func.lower():
                self.pattern_candidates["torch_ascend"].add(func)
            elif "distributed" in func.lower() or "hccl" in func.lower():
                self.pattern_candidates["distributed"].add(func)

    def process_chains(self, chains: List[HostDeviceChain]) -> None:
        """批量处理调用链"""
        for chain in chains:
            self.process_chain(chain)

    def discover_patterns(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        发现高频堆栈模式

        Returns:
            {source_type: [(pattern, frequency), ...]}
        """
        discovered = {}

        for source_type, patterns in self.pattern_candidates.items():
            pattern_freq = [
                (p, self.function_counter[p])
                for p in patterns
                if self.function_counter[p] >= self.min_frequency
            ]
            if pattern_freq:
                pattern_freq.sort(key=lambda x: x[1], reverse=True)
                discovered[source_type] = pattern_freq

        # 也包含高频但未分类的函数
        unclassified = [
            (f, c) for f, c in self.function_counter.most_common(50)
            if c >= self.min_frequency
            and not any(f in patterns for patterns in self.pattern_candidates.values())
        ]
        if unclassified:
            discovered["unclassified"] = unclassified[:20]

        return discovered

    def get_suggested_additions(self) -> Dict[str, List[str]]:
        """
        获取建议添加到 EXTENDED_STACK_PATTERNS 的新模式

        Returns:
            {source_type: [suggested_patterns]}
        """
        discovered = self.discover_patterns()
        suggestions = {}

        for source_type, patterns in discovered.items():
            if source_type != "unclassified":
                # 只建议高频模式
                suggestions[source_type] = [p for p, c in patterns[:5]]

        return suggestions


def classify_operators(
    chains: List[HostDeviceChain],
) -> Tuple[CorrelationStats, List[ClassificationResult]]:
    """
    便捷函数：批量分类算子

    Args:
        chains: HostDeviceChain 列表

    Returns:
        (stats, results)
    """
    classifier = OperatorSourceClassifier()
    results = classifier.classify_batch(chains)
    stats = classifier.build_stats(chains)
    return stats, results


def discover_new_patterns(
    chains: List[HostDeviceChain],
    min_frequency: int = 5,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    便捷函数：发现新的堆栈模式

    Args:
        chains: HostDeviceChain 列表
        min_frequency: 最小频率阈值

    Returns:
        发现的模式
    """
    discovery = StackPatternDiscovery(min_frequency=min_frequency)
    discovery.process_chains(chains)
    return discovery.discover_patterns()
