"""
根因推理引擎

基于 Markdown Skills 的根因推理引擎，用于分析 Profiling 性能问题根因。

支持两种分析模式：
1. 单版本分析 (analyze) - 分析性能/MFU 低的原因
2. 对比分析 (compare) - 对比两个版本的差异，推理性能变化原因
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict

from src.data_loader.stack_types import HostDeviceChain, SourceAnalysisResult, CorrelationStats
from src.analyzers.profiling_diff import ProfilingDiff, OperatorChange, OperatorDiff
from src.analyzers.operator_source_classifier import SourceChange, OperatorSourceClassifier

logger = logging.getLogger(__name__)


@dataclass
class RootCauseRule:
    """根因推理规则"""
    name: str
    trigger_conditions: Dict[str, Any]
    root_cause: str
    evidence_patterns: List[str]
    optimization_suggestions: List[str]
    impact: Dict[str, str]  # performance_impact, fix_difficulty, priority
    rule_file: str = ""


@dataclass
class RootCauseFinding:
    """根因发现"""
    rule_name: str
    root_cause: str
    evidence: List[str]
    affected_operators: List[str]
    optimization_suggestions: List[str]
    impact: str  # "high" / "medium" / "low"
    priority: str  # "P0" / "P1" / "P2"
    confidence: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "affected_operators": self.affected_operators,
            "optimization_suggestions": self.optimization_suggestions,
            "impact": self.impact,
            "priority": self.priority,
            "confidence": self.confidence,
        }

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = [
            f"### [{self.priority}] {self.rule_name}",
            "",
            f"**根因**: {self.root_cause}",
            "",
            "**证据**:",
        ]
        for e in self.evidence:
            lines.append(f"- {e}")

        if self.affected_operators:
            lines.append("")
            lines.append(f"**受影响算子**: {', '.join(self.affected_operators[:10])}")

        lines.append("")
        lines.append("**建议**:")
        for s in self.optimization_suggestions[:5]:
            lines.append(f"- {s}")

        return "\n".join(lines)


class RootCauseSkillEngine:
    """
    基于 Markdown Skills 的根因推理引擎

    Usage:
        engine = RootCauseSkillEngine()
        findings = engine.analyze(chains_a, chains_b, diff_result)
    """

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Args:
            skills_dir: Skills 目录路径
        """
        if skills_dir is None:
            skills_dir = str(Path(__file__).parent.parent / "skills" / "root_cause_analysis")

        self.skills_dir = Path(skills_dir)
        self.rules = self._load_builtin_rules()
        self.classifier = OperatorSourceClassifier()

    def _load_builtin_rules(self) -> List[RootCauseRule]:
        """加载内置规则"""
        rules = []

        # ==================== 对比分析规则 ====================

        # 规则 1: torch.compile 融合问题
        rules.append(RootCauseRule(
            name="torch_compile_fusion_issue",
            trigger_conditions={
                "stack_pattern": ["CompiledFunctionBackward", "torch._inductor"],
                "missing_fusion": ["NPUGroupedMatmul", "FlashAttention"],
                "small_op_increase": 50,  # > 50%
            },
            root_cause="torch.compile 图模式未开启对应的算子融合，导致下发大量小算子",
            evidence_patterns=[
                "堆栈中出现 CompiledFunctionBackward 但缺少 NPUGroupedGMM",
                "小算子数量显著增加",
                "端到端耗时增加但单个算子耗时无明显变化",
            ],
            optimization_suggestions=[
                "启用 torch.compile(model, mode=\"reduce-overhead\")",
                "添加自定义融合规则",
                "检查 mindspeed 配置是否与 torch.compile 兼容",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "medium", "priority": "P0"},
            rule_file="compile_fusion_issue.md",
        ))

        # 规则 2: eager/compile 模式切换
        rules.append(RootCauseRule(
            name="eager_compile_switch",
            trigger_conditions={
                "source_change": True,
                "mode_switch": ["eager", "torch_compile"],
            },
            root_cause="两个 Profiling 版本使用了不同的执行模式（eager vs torch.compile）",
            evidence_patterns=[
                "版本 A 堆栈显示 aten:: 开头的原生算子",
                "版本 B 堆栈显示 CompiledFunction 或 torch._dynamo",
                "两个版本的算子数量差异大",
            ],
            optimization_suggestions=[
                "确保对比的两个版本使用相同的执行模式",
                "分离对比维度，单独分析编译开销和收益",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "low", "priority": "P0"},
            rule_file="eager_compile_switch.md",
        ))

        # 规则 3: 通信瓶颈
        rules.append(RootCauseRule(
            name="communication_bottleneck",
            trigger_conditions={
                "comm_ratio_increase": 10,  # > 10%
                "comm_time_increase": 20,   # > 20%
            },
            root_cause="分布式训练中的通信瓶颈导致性能下降",
            evidence_patterns=[
                "通信时间占比显著增加",
                "未掩盖通信时间增加",
                "存在带宽利用率低于阈值的链路",
            ],
            optimization_suggestions=[
                "优化通信掩盖，使用异步通信",
                "检查网络拓扑和 HCCS/RDMA 配置",
                "根据模型大小选择合适的集合通信算法",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "medium", "priority": "P0"},
            rule_file="communication_bottleneck.md",
        ))

        # ==================== 单版本分析规则 ====================

        # 规则 4: 小算子过多
        rules.append(RootCauseRule(
            name="excessive_small_operators",
            trigger_conditions={
                "small_op_ratio": 30,  # 小算子占比 > 30%
                "small_op_threshold_us": 10,
            },
            root_cause="大量小算子导致 kernel launch 开销过高，GPU/NPU 利用率低",
            evidence_patterns=[
                "大量耗时 <10us 的小算子",
                "算子融合机会未被利用",
                "kernel launch 开销占比较高",
            ],
            optimization_suggestions=[
                "启用算子融合优化（torch.compile 或自定义融合）",
                "使用 torch.jit.script 或 torch.compile 编译模型",
                "检查是否有不必要的张量操作（如多次 .cpu()/.cuda()）",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "medium", "priority": "P0"},
            rule_file="excessive_small_operators.md",
        ))

        # 规则 5: 混合执行模式
        rules.append(RootCauseRule(
            name="mixed_execution_mode",
            trigger_conditions={
                "eager_ratio_min": 10,  # eager 占比 > 10%
                "compile_ratio_min": 10,  # compile 占比 > 10%
            },
            root_cause="eager 和 torch.compile 混合使用导致性能不稳定",
            evidence_patterns=[
                "部分算子使用 eager 模式",
                "部分算子使用 torch.compile 模式",
                "可能存在 graph break 导致编译中断",
            ],
            optimization_suggestions=[
                "检查 torch.compile 的 graph break 点",
                "使用 torch._dynamo.explain() 分析编译失败原因",
                "对不支持编译的算子添加自定义支持",
            ],
            impact={"performance_impact": "medium", "fix_difficulty": "medium", "priority": "P1"},
            rule_file="mixed_execution_mode.md",
        ))

        # 规则 6: 融合算子使用不足
        rules.append(RootCauseRule(
            name="insufficient_fusion_operators",
            trigger_conditions={
                "fusion_op_ratio_max": 10,  # 融合算子占比 < 10%
                "min_total_chains": 50,
            },
            root_cause="融合算子使用率低，存在大量可融合的独立算子",
            evidence_patterns=[
                "融合算子（如 NPUGroupedMatmul, FlashAttention）使用较少",
                "存在可融合的连续算子模式",
                "计算密集型算子被拆分成多个小算子",
            ],
            optimization_suggestions=[
                "启用 mindspeed 或 megatron 的融合算子",
                "使用 FlashAttention 替代标准 Attention",
                "检查是否可以使用 GroupedMatmul 替代多个独立 Matmul",
            ],
            impact={"performance_impact": "medium", "fix_difficulty": "medium", "priority": "P1"},
            rule_file="insufficient_fusion_operators.md",
        ))

        # 规则 7: 通信占比过高
        rules.append(RootCauseRule(
            name="high_communication_ratio",
            trigger_conditions={
                "comm_ratio_threshold": 30,  # 通信占比 > 30%
            },
            root_cause="通信时间占比过高，计算通信比不理想",
            evidence_patterns=[
                "通信时间占总时间比例较高",
                "可能存在通信-计算重叠不足",
                "集群规模扩展效率低",
            ],
            optimization_suggestions=[
                "优化通信掩盖策略（overlap compute and communication）",
                "检查是否可以使用梯度累积减少通信频率",
                "评估是否可以调整并行策略（TP/PP/DP 配置）",
            ],
            impact={"performance_impact": "high", "fix_difficulty": "high", "priority": "P0"},
            rule_file="high_communication_ratio.md",
        ))

        # 规则 8: 内存层次利用不佳
        rules.append(RootCauseRule(
            name="poor_memory_hierarchy_utilization",
            trigger_conditions={
                "small_op_types": ["Add", "Mul", "Div", "Copy", "Contiguous"],
                "small_op_ratio": 20,
            },
            root_cause="内存操作算子过多，内存带宽利用率低",
            evidence_patterns=[
                "大量 Add、Mul、Copy 等内存密集型算子",
                "可能存在不必要的张量拷贝",
                "内存布局不连续导致额外拷贝",
            ],
            optimization_suggestions=[
                "检查张量是否需要 .contiguous() 调用",
                "使用 in-place 操作减少内存分配",
                "优化数据布局（NHWC vs NCHW）",
            ],
            impact={"performance_impact": "medium", "fix_difficulty": "medium", "priority": "P1"},
            rule_file="poor_memory_hierarchy_utilization.md",
        ))

        return rules

    def analyze(
        self,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: Optional[List[SourceChange]] = None,
    ) -> List[RootCauseFinding]:
        """
        执行根因推理

        Args:
            chains_a: 版本 A 的调用链
            chains_b: 版本 B 的调用链
            diff_result: ProfilingDiff 对比结果
            source_changes: 来源变化列表

        Returns:
            RootCauseFinding 列表
        """
        findings = []

        # 1. 分析来源变化
        if source_changes is None:
            source_changes = self.classifier.compute_source_changes(chains_a, chains_b)

        # 2. 检查各规则
        for rule in self.rules:
            finding = self._check_rule(rule, chains_a, chains_b, diff_result, source_changes)
            if finding:
                findings.append(finding)

        # 3. 按优先级排序
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        findings.sort(key=lambda f: (priority_order.get(f.priority, 99), -f.confidence))

        return findings

    def analyze_single(
        self,
        chains: List[HostDeviceChain],
        stats: Optional[CorrelationStats] = None,
        profiling_summary: Optional[Any] = None,
        mfu_metrics: Optional[Any] = None,
    ) -> List[RootCauseFinding]:
        """
        单版本根因推理 - 分析性能/MFU 低的原因

        Args:
            chains: Host-Device 调用链
            stats: 关联统计
            profiling_summary: Profiling 摘要
            mfu_metrics: MFU 指标

        Returns:
            RootCauseFinding 列表
        """
        findings = []

        if not chains:
            logger.warning("No chains provided for single-version root cause analysis")
            return findings

        # 获取统计信息
        if stats is None:
            stats = self.classifier.build_stats(chains)

        total_chains = len(chains)

        # 1. 检查小算子过多问题
        finding = self._check_excessive_small_operators(chains, stats)
        if finding:
            findings.append(finding)

        # 2. 检查混合执行模式
        finding = self._check_mixed_execution_mode(chains, stats)
        if finding:
            findings.append(finding)

        # 3. 检查融合算子使用不足
        finding = self._check_insufficient_fusion_operators(chains, stats)
        if finding:
            findings.append(finding)

        # 4. 检查通信占比过高
        finding = self._check_high_communication_ratio(chains, stats, profiling_summary)
        if finding:
            findings.append(finding)

        # 5. 检查内存层次利用不佳
        finding = self._check_poor_memory_hierarchy(chains, stats)
        if finding:
            findings.append(finding)

        # 6. 根据 MFU 指标补充分析
        if mfu_metrics and hasattr(mfu_metrics, 'overall_mfu'):
            if mfu_metrics.overall_mfu < 0.3:  # MFU < 30%
                finding = self._analyze_low_mfu_causes(chains, stats, mfu_metrics)
                if finding:
                    findings.append(finding)

        # 按优先级排序
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        findings.sort(key=lambda f: (priority_order.get(f.priority, 99), -f.confidence))

        return findings

    def _check_excessive_small_operators(
        self,
        chains: List[HostDeviceChain],
        stats: CorrelationStats,
    ) -> Optional[RootCauseFinding]:
        """检查小算子过多问题"""
        small_op_threshold_us = 10  # 10us
        small_op_ratio_threshold = 30  # 30%

        # 统计小算子
        small_ops = [c for c in chains if c.device_op_dur < small_op_threshold_us]
        small_op_ratio = len(small_ops) / len(chains) * 100 if chains else 0

        if small_op_ratio < small_op_ratio_threshold:
            return None

        # 收集证据
        evidence = [
            f"小算子（<{small_op_threshold_us}us）占比: {small_op_ratio:.1f}%",
            f"小算子数量: {len(small_ops)} / {len(chains)}",
        ]

        # 分析小算子类型
        small_op_types = Counter(c.device_op_name for c in small_ops)
        top_types = small_op_types.most_common(5)
        if top_types:
            evidence.append(f"主要小算子类型: {', '.join(f'{t[0]}({t[1]})' for t in top_types)}")

        # 分析来源
        small_by_source = Counter(c.source_type for c in small_ops)
        if small_by_source:
            top_source = small_by_source.most_common(1)[0]
            evidence.append(f"小算子主要来源: {top_source[0]} ({top_source[1]}个)")

        affected_ops = list(set(c.device_op_name for c in small_ops[:50]))[:20]

        return RootCauseFinding(
            rule_name="excessive_small_operators",
            root_cause="大量小算子导致 kernel launch 开销过高，GPU/NPU 利用率低",
            evidence=evidence,
            affected_operators=affected_ops,
            optimization_suggestions=[
                "启用算子融合优化（torch.compile 或自定义融合）",
                "使用 torch.jit.script 或 torch.compile 编译模型",
                "检查是否有不必要的张量操作（如多次 .cpu()/.cuda()）",
                "考虑使用 fused Adam 或 fused LayerNorm",
            ],
            impact="high",
            priority="P0",
            confidence=min(0.5 + small_op_ratio / 100, 0.95),
        )

    def _check_mixed_execution_mode(
        self,
        chains: List[HostDeviceChain],
        stats: CorrelationStats,
    ) -> Optional[RootCauseFinding]:
        """检查混合执行模式问题"""
        eager_ratio_min = 10
        compile_ratio_min = 10

        by_source = stats.by_source_type
        total = stats.total_chains or 1

        eager_ratio = by_source.get("eager", 0) / total * 100
        compile_ratio = by_source.get("torch_compile", 0) / total * 100

        if eager_ratio < eager_ratio_min or compile_ratio < compile_ratio_min:
            return None

        evidence = [
            f"eager 模式算子: {eager_ratio:.1f}%",
            f"torch.compile 模式算子: {compile_ratio:.1f}%",
            "检测到混合执行模式，可能存在 graph break",
        ]

        # 分析哪些算子在 eager 模式
        eager_ops = [c.device_op_name for c in chains if c.source_type == "eager"]
        compile_ops = [c.device_op_name for c in chains if c.source_type == "torch_compile"]

        if eager_ops:
            top_eager = Counter(eager_ops).most_common(5)
            evidence.append(f"eager 模式主要算子: {', '.join(f'{t[0]}' for t in top_eager)}")

        return RootCauseFinding(
            rule_name="mixed_execution_mode",
            root_cause="eager 和 torch.compile 混合使用导致性能不稳定",
            evidence=evidence,
            affected_operators=list(set(eager_ops + compile_ops))[:20],
            optimization_suggestions=[
                "检查 torch.compile 的 graph break 点",
                "使用 torch._dynamo.explain() 分析编译失败原因",
                "对不支持编译的算子添加自定义支持",
                "考虑使用 torch.compile 的 fullgraph=True 选项",
            ],
            impact="medium",
            priority="P1",
            confidence=0.7,
        )

    def _check_insufficient_fusion_operators(
        self,
        chains: List[HostDeviceChain],
        stats: CorrelationStats,
    ) -> Optional[RootCauseFinding]:
        """检查融合算子使用不足"""
        fusion_ratio_max = 10
        min_total_chains = 50

        if len(chains) < min_total_chains:
            return None

        by_source = stats.by_source_type
        total = stats.total_chains or 1
        fusion_ratio = by_source.get("fusion_op", 0) / total * 100

        if fusion_ratio >= fusion_ratio_max:
            return None

        evidence = [
            f"融合算子占比: {fusion_ratio:.1f}%（建议 > 10%）",
            f"融合算子数量: {by_source.get('fusion_op', 0)}",
        ]

        # 检查是否存在可融合的算子模式
        op_names = [c.device_op_name for c in chains]
        op_counter = Counter(op_names)

        # 检查可能的融合机会
        fusion_candidates = []
        for op_name, count in op_counter.most_common(20):
            if any(p in op_name for p in ["Matmul", "Linear", "Add", "Mul", "LayerNorm", "Softmax"]):
                fusion_candidates.append(f"{op_name}({count})")

        if fusion_candidates:
            evidence.append(f"可能存在融合机会的算子: {', '.join(fusion_candidates[:5])}")

        # 获取实际使用的融合算子
        fusion_ops = [c.device_op_name for c in chains if c.source_type == "fusion_op"]
        if fusion_ops:
            evidence.append(f"当前使用的融合算子: {', '.join(set(fusion_ops[:5]))}")

        return RootCauseFinding(
            rule_name="insufficient_fusion_operators",
            root_cause="融合算子使用率低，存在大量可融合的独立算子",
            evidence=evidence,
            affected_operators=fusion_candidates[:10],
            optimization_suggestions=[
                "启用 mindspeed 或 megatron 的融合算子",
                "使用 FlashAttention 替代标准 Attention",
                "检查是否可以使用 GroupedMatmul 替代多个独立 Matmul",
                "考虑使用 fused AdamW 优化器",
            ],
            impact="medium",
            priority="P1",
            confidence=0.6,
        )

    def _check_high_communication_ratio(
        self,
        chains: List[HostDeviceChain],
        stats: CorrelationStats,
        profiling_summary: Optional[Any],
    ) -> Optional[RootCauseFinding]:
        """检查通信占比过高"""
        comm_ratio_threshold = 30

        by_source = stats.by_source_type
        total = stats.total_chains or 1
        comm_ratio = by_source.get("distributed", 0) / total * 100

        # 也从 profiling_summary 获取通信占比
        if profiling_summary and hasattr(profiling_summary, 'avg_comm_time'):
            total_time = (profiling_summary.avg_compute_time +
                         profiling_summary.avg_comm_time +
                         profiling_summary.avg_free_time)
            if total_time > 0:
                actual_comm_ratio = profiling_summary.avg_comm_time / total_time * 100
                if actual_comm_ratio > comm_ratio_threshold:
                    comm_ratio = actual_comm_ratio

        if comm_ratio < comm_ratio_threshold:
            return None

        evidence = [
            f"通信算子占比: {comm_ratio:.1f}%",
        ]

        # 分析通信算子类型
        comm_chains = [c for c in chains if c.source_type == "distributed"]
        if comm_chains:
            comm_ops = Counter(c.device_op_name for c in comm_chains)
            top_comm = comm_ops.most_common(5)
            evidence.append(f"主要通信算子: {', '.join(f'{t[0]}({t[1]})' for t in top_comm)}")

        return RootCauseFinding(
            rule_name="high_communication_ratio",
            root_cause="通信时间占比过高，计算通信比不理想",
            evidence=evidence,
            affected_operators=list(set(c.device_op_name for c in comm_chains))[:10],
            optimization_suggestions=[
                "优化通信掩盖策略（overlap compute and communication）",
                "检查是否可以使用梯度累积减少通信频率",
                "评估是否可以调整并行策略（TP/PP/DP 配置）",
                "检查 HCCL 集合通信配置是否最优",
            ],
            impact="high",
            priority="P0",
            confidence=0.7,
        )

    def _check_poor_memory_hierarchy(
        self,
        chains: List[HostDeviceChain],
        stats: CorrelationStats,
    ) -> Optional[RootCauseFinding]:
        """检查内存层次利用不佳"""
        small_op_ratio_threshold = 20
        memory_op_types = ["Add", "Mul", "Div", "Copy", "Contiguous", "Transpose", "Reshape"]

        # 统计内存操作算子
        memory_ops = [c for c in chains
                     if any(t in c.device_op_name for t in memory_op_types)]
        memory_op_ratio = len(memory_ops) / len(chains) * 100 if chains else 0

        if memory_op_ratio < small_op_ratio_threshold:
            return None

        evidence = [
            f"内存操作算子占比: {memory_op_ratio:.1f}%",
            f"内存操作算子数量: {len(memory_ops)}",
        ]

        # 分析内存操作类型
        mem_op_types = Counter(c.device_op_name for c in memory_ops)
        top_types = mem_op_types.most_common(5)
        if top_types:
            evidence.append(f"主要内存操作: {', '.join(f'{t[0]}({t[1]})' for t in top_types)}")

        # 检查 Copy 算子数量（可能有不必要的拷贝）
        copy_ops = [c for c in chains if "Copy" in c.device_op_name]
        if len(copy_ops) > len(chains) * 0.1:
            evidence.append(f"检测到大量 Copy 算子: {len(copy_ops)}，可能存在不必要的拷贝")

        return RootCauseFinding(
            rule_name="poor_memory_hierarchy_utilization",
            root_cause="内存操作算子过多，内存带宽利用率低",
            evidence=evidence,
            affected_operators=list(set(c.device_op_name for c in memory_ops))[:20],
            optimization_suggestions=[
                "检查张量是否需要 .contiguous() 调用",
                "使用 in-place 操作减少内存分配",
                "优化数据布局（NHWC vs NCHW）",
                "减少 CPU-GPU 之间的数据传输",
            ],
            impact="medium",
            priority="P1",
            confidence=0.6,
        )

    def _analyze_low_mfu_causes(
        self,
        chains: List[HostDeviceChain],
        stats: CorrelationStats,
        mfu_metrics: Any,
    ) -> Optional[RootCauseFinding]:
        """分析 MFU 低的原因"""
        mfu = mfu_metrics.overall_mfu if hasattr(mfu_metrics, 'overall_mfu') else 0

        evidence = [
            f"MFU 仅为 {mfu*100:.1f}%，远低于理想值（>50%）",
        ]

        # 分析可能的原因
        by_source = stats.by_source_type
        total = stats.total_chains or 1

        # 检查计算密集型算子占比
        compute_ops = by_source.get("fusion_op", 0) + by_source.get("mindspeed", 0)
        compute_ratio = compute_ops / total * 100

        if compute_ratio < 20:
            evidence.append(f"计算密集型算子占比仅 {compute_ratio:.1f}%，计算效率低")

        # 检查小算子占比
        small_ops = [c for c in chains if c.device_op_dur < 10]
        small_ratio = len(small_ops) / total * 100
        if small_ratio > 30:
            evidence.append(f"小算子占比 {small_ratio:.1f}%，kernel launch 开销大")

        return RootCauseFinding(
            rule_name="low_mfu_analysis",
            root_cause="MFU 偏低，综合分析发现多个性能瓶颈",
            evidence=evidence,
            affected_operators=[],
            optimization_suggestions=[
                "优先解决小算子过多和融合算子不足的问题",
                "优化并行策略以提高计算占比",
                "检查是否存在不必要的同步点",
            ],
            impact="high",
            priority="P0",
            confidence=0.5,
        )

    def _check_rule(
        self,
        rule: RootCauseRule,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: List[SourceChange],
    ) -> Optional[RootCauseFinding]:
        """检查单个规则是否匹配"""
        conditions = rule.trigger_conditions

        # 检查 torch.compile 融合问题
        if rule.name == "torch_compile_fusion_issue":
            return self._check_compile_fusion_issue(rule, chains_a, chains_b, diff_result, source_changes)

        # 检查 eager/compile 模式切换
        if rule.name == "eager_compile_switch":
            return self._check_eager_compile_switch(rule, chains_a, chains_b, diff_result, source_changes)

        # 检查通信瓶颈
        if rule.name == "communication_bottleneck":
            return self._check_communication_bottleneck(rule, diff_result)

        return None

    def _check_compile_fusion_issue(
        self,
        rule: RootCauseRule,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: List[SourceChange],
    ) -> Optional[RootCauseFinding]:
        """检查 torch.compile 融合问题"""
        conditions = rule.trigger_conditions

        # 统计版本 B 中的 torch.compile 算子
        compile_chains_b = [c for c in chains_b if c.source_type == "torch_compile"]
        fusion_chains_a = [c for c in chains_a if c.source_type == "fusion_op"]

        if not compile_chains_b:
            return None

        # 检查小算子增加
        small_op_increase_threshold = conditions.get("small_op_increase", 50)

        # 统计小算子数量
        small_ops_a = [c for c in chains_a if c.device_op_dur < 10]  # < 10us
        small_ops_b = [c for c in chains_b if c.device_op_dur < 10]

        small_op_increase = 0
        if len(small_ops_a) > 0:
            small_op_increase = (len(small_ops_b) - len(small_ops_a)) / len(small_ops_a) * 100

        if small_op_increase < small_op_increase_threshold:
            return None

        # 收集证据
        evidence = []
        affected_ops = []

        # 检查新增的小算子
        if diff_result.operator_diff and diff_result.operator_diff.new_operators:
            new_small_ops = [op for op in diff_result.operator_diff.new_operators
                           if "Add" in op.name or "zeroslike" in op.name or "Mul" in op.name]
            if new_small_ops:
                evidence.append(f"新增小算子数量: {len(new_small_ops)}")
                affected_ops = [op.name for op in new_small_ops[:10]]

        # 检查堆栈模式
        compile_pattern_count = sum(1 for c in compile_chains_b
                                   if any(p in " ".join(c.python_stack) for p in conditions.get("stack_pattern", [])))
        if compile_pattern_count > 0:
            evidence.append(f"检测到 {compile_pattern_count} 个 torch.compile 编译算子")

        # 检查缺少的融合算子
        fusion_ops_a = set(c.device_op_name for c in fusion_chains_a)
        fusion_ops_b = set(c.device_op_name for c in chains_b if c.source_type == "fusion_op")
        missing_fusion = fusion_ops_a - fusion_ops_b
        if missing_fusion:
            evidence.append(f"版本 B 缺少融合算子: {', '.join(list(missing_fusion)[:5])}")

        if not evidence:
            return None

        return RootCauseFinding(
            rule_name=rule.name,
            root_cause=rule.root_cause,
            evidence=evidence,
            affected_operators=affected_ops,
            optimization_suggestions=rule.optimization_suggestions,
            impact=rule.impact.get("performance_impact", "medium"),
            priority=rule.impact.get("priority", "P1"),
            confidence=min(0.5 + small_op_increase / 100, 0.95),
        )

    def _check_eager_compile_switch(
        self,
        rule: RootCauseRule,
        chains_a: List[HostDeviceChain],
        chains_b: List[HostDeviceChain],
        diff_result: ProfilingDiff,
        source_changes: List[SourceChange],
    ) -> Optional[RootCauseFinding]:
        """检查 eager/compile 模式切换"""
        # 统计来源分布
        sources_a = Counter(c.source_type for c in chains_a)
        sources_b = Counter(c.source_type for c in chains_b)

        # 检查模式切换
        mode_switches = [sc for sc in source_changes if sc.change_type == "mode_switch"]

        if not mode_switches:
            # 检查整体模式变化
            eager_ratio_a = sources_a.get("eager", 0) / len(chains_a) if chains_a else 0
            eager_ratio_b = sources_b.get("eager", 0) / len(chains_b) if chains_b else 0
            compile_ratio_a = sources_a.get("torch_compile", 0) / len(chains_a) if chains_a else 0
            compile_ratio_b = sources_b.get("torch_compile", 0) / len(chains_b) if chains_b else 0

            if abs(eager_ratio_a - eager_ratio_b) < 0.3 and abs(compile_ratio_a - compile_ratio_b) < 0.3:
                return None

        evidence = []
        affected_ops = [sc.operator_name for sc in mode_switches[:20]]

        if sources_a.get("eager", 0) > 0 and sources_b.get("torch_compile", 0) > 0:
            evidence.append("版本 A 主要使用 eager 模式")
            evidence.append("版本 B 主要使用 torch.compile 模式")
        elif sources_a.get("torch_compile", 0) > 0 and sources_b.get("eager", 0) > 0:
            evidence.append("版本 A 主要使用 torch.compile 模式")
            evidence.append("版本 B 主要使用 eager 模式")

        if mode_switches:
            evidence.append(f"检测到 {len(mode_switches)} 个算子发生模式切换")

        if not evidence:
            return None

        return RootCauseFinding(
            rule_name=rule.name,
            root_cause=rule.root_cause,
            evidence=evidence,
            affected_operators=affected_ops,
            optimization_suggestions=rule.optimization_suggestions,
            impact=rule.impact.get("performance_impact", "medium"),
            priority=rule.impact.get("priority", "P0"),
            confidence=0.8,
        )

    def _check_communication_bottleneck(
        self,
        rule: RootCauseRule,
        diff_result: ProfilingDiff,
    ) -> Optional[RootCauseFinding]:
        """检查通信瓶颈"""
        conditions = rule.trigger_conditions

        # 检查通信相关指标
        comm_diff = diff_result.comm_diff
        if not comm_diff:
            return None

        evidence = []
        comm_ratio_increase_threshold = conditions.get("comm_ratio_increase", 10)

        # 检查通信时间变化
        if comm_diff.total_comm_time_change:
            change_pct = abs(comm_diff.total_comm_time_change.change_pct)
            if change_pct > conditions.get("comm_time_increase", 20):
                evidence.append(f"通信总时间变化: {change_pct:.1f}%")

        # 检查通信占比变化
        if diff_result.summary_diff.comm_ratio:
            comm_ratio_change = abs(diff_result.summary_diff.comm_ratio.change_pct)
            if comm_ratio_change > comm_ratio_increase_threshold:
                evidence.append(f"通信占比变化: {comm_ratio_change:.1f}%")

        # 检查通信掩盖率变化
        if comm_diff.overlap_ratio_change:
            overlap_change = comm_diff.overlap_ratio_change.change_pct
            if overlap_change < -conditions.get("overlap_ratio_decrease", 15):
                evidence.append(f"通信掩盖率下降: {abs(overlap_change):.1f}%")

        if not evidence:
            return None

        return RootCauseFinding(
            rule_name=rule.name,
            root_cause=rule.root_cause,
            evidence=evidence,
            affected_operators=[],
            optimization_suggestions=rule.optimization_suggestions,
            impact=rule.impact.get("performance_impact", "high"),
            priority=rule.impact.get("priority", "P0"),
            confidence=0.7,
        )


def analyze_root_causes(
    chains_a: List[HostDeviceChain],
    chains_b: List[HostDeviceChain],
    diff_result: ProfilingDiff,
) -> List[RootCauseFinding]:
    """
    便捷函数：执行根因分析

    Args:
        chains_a: 版本 A 的调用链
        chains_b: 版本 B 的调用链
        diff_result: ProfilingDiff 对比结果

    Returns:
        RootCauseFinding 列表
    """
    engine = RootCauseSkillEngine()
    return engine.analyze(chains_a, chains_b, diff_result)
