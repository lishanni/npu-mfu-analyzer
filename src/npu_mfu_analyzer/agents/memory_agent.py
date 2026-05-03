"""
Memory Agent

分析内存使用，识别峰值、碎片、泄漏和 OOM 风险。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

import pandas as pd

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AnalysisResult
from npu_mfu_analyzer.llm.llm_interface import LLMInterface
from npu_mfu_analyzer.llm.prompts import MEMORY_ANALYSIS_SYSTEM

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """内存分析指标"""
    # 峰值内存（字节）
    peak_memory_bytes: float = 0.0
    peak_memory_mb: float = 0.0
    peak_memory_gb: float = 0.0

    # 内存使用分布
    model_memory_mb: float = 0.0       # 模型参数
    optimizer_memory_mb: float = 0.0   # 优化器状态
    activation_memory_mb: float = 0.0  # 激活值
    gradient_memory_mb: float = 0.0    # 梯度
    temp_memory_mb: float = 0.0        # 临时内存

    # 内存碎片
    fragmentation_ratio: float = 0.0   # 碎片率 (%)
    fragmentation_severity: str = "low"  # low, medium, high
    free_holes: int = 0               # 碎片空洞数量
    avg_free_block_kb: float = 0.0    # 平均空闲块大小

    # 内存泄漏检测
    has_leak: bool = False
    leak_rate_mb_per_step: float = 0.0  # 每步泄漏量（MB）
    leak_confidence: float = 0.0        # 泄漏置信度 (0-1)

    # 分配模式
    avg_allocation_kb: float = 0.0
    allocation_count: int = 0
    small_allocation_ratio: float = 0.0  # 小分配（<1KB）占比
    large_allocation_ratio: float = 0.0   # 大分配（>1MB）占比

    # 设备信息
    device_memory_gb: float = 64.0     # 设备总内存（默认 64GB）
    memory_utilization: float = 0.0    # 内存利用率 (%)

    # 风险评估
    oom_risk: str = "low"  # low, medium, high
    oom_risk_factors: List[str] = field(default_factory=list)

    def calculate_utilization(self):
        """计算内存利用率"""
        if self.device_memory_gb > 0:
            self.memory_utilization = (self.peak_memory_gb / self.device_memory_gb) * 100

            # OOM 风险评估（考虑多个因素）
            risk_factors = []
            risk_score = 0

            # 利用率风险
            if self.memory_utilization > 95:
                risk_score += 3
                risk_factors.append("极高内存利用率（>95%）")
            elif self.memory_utilization > 85:
                risk_score += 2
                risk_factors.append("高内存利用率（>85%）")
            elif self.memory_utilization > 75:
                risk_score += 1
                risk_factors.append("中等内存利用率（>75%）")

            # 碎片化风险
            if self.fragmentation_ratio > 30:
                risk_score += 2
                risk_factors.append(f"严重内存碎片（{self.fragmentation_ratio:.1f}%）")
            elif self.fragmentation_ratio > 20:
                risk_score += 1
                risk_factors.append(f"显著内存碎片（{self.fragmentation_ratio:.1f}%）")

            # 泄漏风险
            if self.has_leak and self.leak_confidence > 0.7:
                risk_score += 3
                risk_factors.append(f"检测到内存泄漏（{self.leak_rate_mb_per_step:.2f} MB/step）")
            elif self.has_leak and self.leak_confidence > 0.5:
                risk_score += 1
                risk_factors.append(f"可能存在内存泄漏")

            # 确定风险等级
            if risk_score >= 5 or self.memory_utilization > 95:
                self.oom_risk = "high"
            elif risk_score >= 2 or self.memory_utilization > 75:
                self.oom_risk = "medium"
            else:
                self.oom_risk = "low"

            self.oom_risk_factors = risk_factors

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 内存分析",
            "",
            "### 内存使用概况",
            f"- 峰值内存: {self.peak_memory_gb:.2f} GB ({self.peak_memory_mb:.0f} MB)",
            f"- 设备总内存: {self.device_memory_gb:.0f} GB",
            f"- 内存利用率: {self.memory_utilization:.1f}%",
            f"- OOM 风险: {self.oom_risk.upper()}",
        ]

        if self.oom_risk_factors:
            lines.append("- 风险因素:")
            for factor in self.oom_risk_factors:
                lines.append(f"  • {factor}")

        lines.append("")
        lines.append("### 内存分布")

        if self.model_memory_mb > 0:
            lines.append(f"- 模型参数: {self.model_memory_mb:.0f} MB")
        if self.optimizer_memory_mb > 0:
            lines.append(f"- 优化器状态: {self.optimizer_memory_mb:.0f} MB")
        if self.activation_memory_mb > 0:
            lines.append(f"- 激活值: {self.activation_memory_mb:.0f} MB")
        if self.gradient_memory_mb > 0:
            lines.append(f"- 梯度: {self.gradient_memory_mb:.0f} MB")
        if self.temp_memory_mb > 0:
            lines.append(f"- 临时内存: {self.temp_memory_mb:.0f} MB")

        if self.fragmentation_ratio > 0 or self.free_holes > 0:
            lines.append("")
            lines.append("### 内存碎片")
            lines.append(f"- 碎片率: {self.fragmentation_ratio:.1f}% ({self.fragmentation_severity.upper()})")
            lines.append(f"- 碎片空洞数: {self.free_holes}")
            if self.avg_free_block_kb > 0:
                lines.append(f"- 平均空闲块: {self.avg_free_block_kb:.0f} KB")

        if self.has_leak:
            lines.append("")
            lines.append("### ⚠️ 内存泄漏检测")
            lines.append(f"- 泄漏率: {self.leak_rate_mb_per_step:.2f} MB/step")
            lines.append(f"- 置信度: {self.leak_confidence:.0%}")

        if self.allocation_count > 0:
            lines.append("")
            lines.append("### 分配模式")
            lines.append(f"- 总分配次数: {self.allocation_count:,}")
            lines.append(f"- 平均分配大小: {self.avg_allocation_kb:.1f} KB")
            lines.append(f"- 小分配占比（<1KB）: {self.small_allocation_ratio:.1%}")
            lines.append(f"- 大分配占比（>1MB）: {self.large_allocation_ratio:.1%}")

        return "\n".join(lines)


@dataclass
class MemoryEvent:
    """内存事件"""
    timestamp: float = 0.0
    operation: str = ""  # "allocate", "free"
    size_bytes: float = 0.0
    total_allocated: float = 0.0
    operator_name: str = ""
    address: int = 0  # 内存地址（用于碎片分析）


@dataclass
class MemoryLeakResult:
    """内存泄漏检测结果"""
    has_leak: bool = False
    leak_rate_mb_per_step: float = 0.0
    confidence: float = 0.0
    trend_slope: float = 0.0  # 趋势斜率
    r_squared: float = 0.0    # 拟合度
    leak_start_step: int = 0  # 泄漏开始步骤
    suspected_allocations: List[str] = field(default_factory=list)


@dataclass
class MemoryFragmentationResult:
    """内存碎片化分析结果"""
    fragmentation_ratio: float = 0.0
    severity: str = "low"
    free_holes: int = 0
    total_free_bytes: float = 0.0
    largest_free_block_kb: float = 0.0
    avg_free_block_kb: float = 0.0
    free_block_count: int = 0


@dataclass
class AllocationPatternResult:
    """分配模式分析结果"""
    allocation_count: int = 0
    avg_allocation_kb: float = 0.0
    median_allocation_kb: float = 0.0
    std_allocation_kb: float = 0.0
    small_allocation_ratio: float = 0.0  # < 1KB
    large_allocation_ratio: float = 0.0  # > 1MB
    tiny_allocation_ratio: float = 0.0   # < 100B
    huge_allocation_ratio: float = 0.0   # > 10MB
    size_distribution: Dict[str, int] = field(default_factory=dict)
    hot_operators: List[Tuple[str, int]] = field(default_factory=list)  # (op_name, count)


@dataclass
class MemoryAnalysisData:
    """内存分析数据"""
    metrics: Optional[MemoryMetrics] = None
    memory_events: List[MemoryEvent] = field(default_factory=list)
    top_memory_operators: List[Dict[str, Any]] = field(default_factory=list)
    memory_timeline: List[Dict[str, float]] = field(default_factory=list)

    # 新增：分析结果
    leak_result: Optional[MemoryLeakResult] = None
    fragmentation_result: Optional[MemoryFragmentationResult] = None
    allocation_pattern: Optional[AllocationPatternResult] = None

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = []

        if self.metrics:
            lines.append(self.metrics.to_prompt_text())

        if self.top_memory_operators:
            lines.append("")
            lines.append("### Top 内存消耗算子")
            for i, op in enumerate(self.top_memory_operators[:10], 1):
                name = op.get("name", "unknown")
                mem_mb = op.get("memory_mb", 0)
                lines.append(f"{i}. {name}: {mem_mb:.0f} MB")

        return "\n".join(lines)


class MemoryAnalyzer:
    """
    内存分析器 - 核心算法实现

    包含：
    1. 内存碎片化检测算法
    2. 内存泄漏检测算法
    3. 分配模式分析算法
    """

    @staticmethod
    def detect_leak(
        memory_timeline: List[float],
        step_numbers: List[int] = None,
        min_steps: int = 10,
        threshold_mb: float = 1.0
    ) -> MemoryLeakResult:
        """
        检测内存泄漏

        算法：
        1. 对内存时间序列进行线性回归
        2. 计算斜率和 R²
        3. 根据斜率和 R² 判断是否存在泄漏

        Args:
            memory_timeline: 内存使用时间序列（MB）
            step_numbers: 对应的步骤编号
            min_steps: 最少步骤数
            threshold_mb: 泄漏阈值（MB/step）

        Returns:
            MemoryLeakResult
        """
        if len(memory_timeline) < min_steps:
            return MemoryLeakResult()

        # 使用 numpy 进行线性回归
        x = np.arange(len(memory_timeline))
        y = np.array(memory_timeline)

        # 移除 NaN
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < min_steps:
            return MemoryLeakResult()

        x = x[valid_mask]
        y = y[valid_mask]

        # 线性回归
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # 斜率（MB/step）

        # 计算 R²
        p = np.poly1d(coeffs)
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 判断是否存在泄漏
        has_leak = False
        confidence = 0.0

        # 正斜率表明内存增长
        if slope > 0:
            # 根据斜率和 R² 计算置信度
            if r_squared > 0.8:
                confidence = min(0.95, 0.7 + r_squared * 0.25)
            elif r_squared > 0.5:
                confidence = 0.5 + r_squared * 0.3

            # 如果斜率超过阈值且置信度足够高
            if slope > threshold_mb and confidence > 0.6:
                has_leak = True

        # 查找泄漏开始点（增长率开始变大的位置）
        leak_start_step = 0
        if has_leak and len(y) > 20:
            # 计算滑动窗口增长率
            window = 5
            growth_rates = []
            for i in range(window, len(y)):
                recent_slope = (y[i] - y[i - window]) / window
                growth_rates.append(recent_slope)

            # 找到增长率持续超过平均值的起始点
            avg_growth = np.mean(growth_rates)
            for i, rate in enumerate(growth_rates):
                if rate > avg_growth * 1.5:
                    leak_start_step = i
                    break

        return MemoryLeakResult(
            has_leak=has_leak,
            leak_rate_mb_per_step=slope,
            confidence=confidence,
            trend_slope=slope,
            r_squared=r_squared,
            leak_start_step=leak_start_step,
        )

    @staticmethod
    def analyze_fragmentation(
        free_blocks: List[Tuple[int, int]],  # (address, size)
        total_memory_bytes: float,
    ) -> MemoryFragmentationResult:
        """
        分析内存碎片化

        算法：
        1. 统计空闲块数量和大小分布
        2. 计算碎片率（无法使用的小块占比）
        3. 评估碎片化严重程度

        Args:
            free_blocks: 空闲块列表 [(address, size_bytes), ...]
            total_memory_bytes: 总内存（字节）

        Returns:
            MemoryFragmentationResult
        """
        if not free_blocks or total_memory_bytes == 0:
            return MemoryFragmentationResult()

        # 按大小排序
        sizes = np.array([size for _, size in free_blocks])

        total_free = np.sum(sizes)
        block_count = len(sizes)

        # 统计不同大小的块
        tiny_blocks = np.sum(sizes < 1024)  # < 1KB
        small_blocks = np.sum(sizes < 1024 * 1024)  # < 1MB
        large_blocks = np.sum(sizes >= 1024 * 1024)  # >= 1MB

        # 计算碎片率
        # 定义：小于 1MB 的块占空闲内存的比例
        fragmented_bytes = np.sum(sizes[sizes < 1024 * 1024])
        fragmentation_ratio = (fragmented_bytes / total_free * 100) if total_free > 0 else 0

        # 评估严重程度
        if fragmentation_ratio > 30 or block_count > 1000:
            severity = "high"
        elif fragmentation_ratio > 15 or block_count > 500:
            severity = "medium"
        else:
            severity = "low"

        # 统计指标
        largest_block = np.max(sizes) if len(sizes) > 0 else 0
        avg_block = np.mean(sizes) if len(sizes) > 0 else 0

        return MemoryFragmentationResult(
            fragmentation_ratio=fragmentation_ratio,
            severity=severity,
            free_holes=block_count,
            total_free_bytes=total_free,
            largest_free_block_kb=largest_block / 1024,
            avg_free_block_kb=avg_block / 1024,
            free_block_count=block_count,
        )

    @staticmethod
    def analyze_allocation_patterns(
        allocations: List[float],  # 分配大小（字节）
    ) -> AllocationPatternResult:
        """
        分析内存分配模式

        算法：
        1. 统计分配大小分布
        2. 识别异常分配模式
        3. 找出热点操作

        Args:
            allocations: 分配大小列表（字节）

        Returns:
            AllocationPatternResult
        """
        if not allocations:
            return AllocationPatternResult()

        sizes = np.array(allocations)
        sizes_kb = sizes / 1024  # 转换为 KB

        count = len(sizes)
        avg_size = np.mean(sizes_kb)
        median_size = np.median(sizes_kb)
        std_size = np.std(sizes_kb)

        # 统计不同大小的分配
        tiny = np.sum(sizes < 100) / count  # < 100B
        small = np.sum(sizes < 1024) / count  # < 1KB
        large = np.sum(sizes > 1024 * 1024) / count  # > 1MB
        huge = np.sum(sizes > 10 * 1024 * 1024) / count  # > 10MB

        # 大小分布直方图
        size_bins = {
            "< 1KB": np.sum(sizes < 1024),
            "1KB - 10KB": np.sum((sizes >= 1024) & (sizes < 10 * 1024)),
            "10KB - 100KB": np.sum((sizes >= 10 * 1024) & (sizes < 100 * 1024)),
            "100KB - 1MB": np.sum((sizes >= 100 * 1024) & (sizes < 1024 * 1024)),
            "1MB - 10MB": np.sum((sizes >= 1024 * 1024) & (sizes < 10 * 1024 * 1024)),
            "> 10MB": np.sum(sizes >= 10 * 1024 * 1024),
        }

        # 检测异常模式
        has_many_tiny = tiny > 0.5  # 超过 50% 是微分配
        has_size_variance = std_size / avg_size > 2.0 if avg_size > 0 else False

        return AllocationPatternResult(
            allocation_count=count,
            avg_allocation_kb=avg_size,
            median_allocation_kb=median_size,
            std_allocation_kb=std_size,
            small_allocation_ratio=small,
            large_allocation_ratio=large,
            tiny_allocation_ratio=tiny,
            huge_allocation_ratio=huge,
            size_distribution=size_bins,
        )


class MemoryAgent(BaseAgent):
    """
    Memory Agent

    功能：
    1. 峰值内存分析
    2. 内存分布分析（模型/优化器/激活值/梯度）
    3. 内存碎片检测（算法实现）
    4. 内存泄漏检测（算法实现）
    5. 分配模式分析（算法实现）
    6. OOM 风险评估（多因素模型）
    7. 优化建议（重计算、梯度累积、混合精度等）
    """

    PROMPT_TEMPLATE = """
你是昇腾 NPU 内存优化专家。分析以下内存使用数据：

{data_summary}

## 分析任务
1. **内存使用评估**：
   - 内存利用率是否合理？
   - 是否存在 OOM 风险？
   - 内存峰值出现在哪个阶段？

2. **内存分布分析**：
   - 各部分内存占比是否正常？
   - 激活值内存是否过大？（大模型常见问题）
   - 是否有内存泄漏迹象？

3. **内存碎片分析**：
   - 碎片率是否过高？（> 20% 需要关注）
   - 是否需要内存整理？

4. **分配模式分析**：
   - 是否有过多的微小分配？（< 1KB）
   - 分配大小分布是否合理？

5. **优化建议**：
   - 激活值重计算（Activation Checkpointing）
   - 梯度累积（Gradient Accumulation）
   - 混合精度训练（AMP）
   - 优化器状态卸载（Optimizer Offload）
   - ZeRO 优化
   - 内存池预分配

请给出详细的分析结论和具体优化建议。
"""

    def __init__(
        self,
        llm: LLMInterface,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="MemoryAgent",
            llm=llm,
            system_prompt=MEMORY_ANALYSIS_SYSTEM,
            config=config
        )
        self.analyzer = MemoryAnalyzer()

    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        分析内存数据

        Args:
            data: 包含以下可选字段：
                - memory_df: 内存数据 DataFrame
                - peak_memory_mb: 峰值内存（MB）
                - memory_events: 内存事件列表
                - memory_timeline: 内存时间序列（用于泄漏检测）
                - free_blocks: 空闲块列表（用于碎片分析）
                - allocations: 分配大小列表（用于模式分析）
                - device_memory_gb: 设备总内存（GB）

        Returns:
            AnalysisResult
        """
        try:
            # 1. 准备分析数据（包含核心算法）
            analysis_data = self._prepare_analysis_data(data)

            # 2. 执行核心分析算法
            self._run_core_algorithms(analysis_data, data)

            # 3. 生成 Prompt 并调用 LLM
            prompt = self.format_prompt(
                self.PROMPT_TEMPLATE,
                data_summary=analysis_data.to_prompt_text()
            )
            response = await self.call_llm(prompt)

            # 4. 构建结果
            metrics = analysis_data.metrics
            peak_gb = metrics.peak_memory_gb if metrics else 0
            oom_risk = metrics.oom_risk if metrics else "unknown"

            details = {
                "peak_memory_gb": peak_gb,
                "peak_memory_mb": metrics.peak_memory_mb if metrics else 0,
                "memory_utilization": metrics.memory_utilization if metrics else 0,
                "oom_risk": oom_risk,
                "model_memory_mb": metrics.model_memory_mb if metrics else 0,
                "optimizer_memory_mb": metrics.optimizer_memory_mb if metrics else 0,
                "activation_memory_mb": metrics.activation_memory_mb if metrics else 0,
                "gradient_memory_mb": metrics.gradient_memory_mb if metrics else 0,
                "training_hints": data.get("training_hints", {}),
            }

            # 添加泄漏检测结果
            if analysis_data.leak_result:
                leak = analysis_data.leak_result
                details["leak_detected"] = leak.has_leak
                details["leak_rate_mb_per_step"] = leak.leak_rate_mb_per_step
                details["leak_confidence"] = leak.confidence

            # 添加碎片检测结果
            if analysis_data.fragmentation_result:
                frag = analysis_data.fragmentation_result
                details["fragmentation_ratio"] = frag.fragmentation_ratio
                details["fragmentation_severity"] = frag.severity
                details["free_holes"] = frag.free_holes

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary=self._build_summary(analysis_data),
                details=details,
                recommendations=self._extract_recommendations(response, analysis_data),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Memory analysis failed: {e}", exc_info=True)
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="内存分析失败",
                error=str(e),
            )

    def _build_summary(self, analysis_data: MemoryAnalysisData) -> str:
        """构建分析摘要"""
        metrics = analysis_data.metrics
        if not metrics:
            return "无有效数据"

        parts = [f"峰值内存: {metrics.peak_memory_gb:.2f} GB"]

        # 添加泄漏信息
        if analysis_data.leak_result and analysis_data.leak_result.has_leak:
            leak = analysis_data.leak_result
            parts.append(f"⚠️ 检测到泄漏 ({leak.leak_rate_mb_per_step:.2f} MB/step)")

        # 添加碎片信息
        if metrics.fragmentation_ratio > 20:
            parts.append(f"碎片率 {metrics.fragmentation_ratio:.0f}%")

        # 添加 OOM 风险
        parts.append(f"OOM 风险: {metrics.oom_risk.upper()}")

        return ", ".join(parts)

    def _prepare_analysis_data(self, data: Dict[str, Any]) -> MemoryAnalysisData:
        """准备分析数据"""
        analysis_data = MemoryAnalysisData()
        metrics = MemoryMetrics()

        # 设备内存
        metrics.device_memory_gb = float(data.get("device_memory_gb", 64.0))

        # 峰值内存
        if "peak_memory_mb" in data:
            metrics.peak_memory_mb = float(data["peak_memory_mb"])
            metrics.peak_memory_gb = metrics.peak_memory_mb / 1024
            metrics.peak_memory_bytes = metrics.peak_memory_mb * 1024 * 1024

        if "peak_memory_bytes" in data:
            metrics.peak_memory_bytes = float(data["peak_memory_bytes"])
            metrics.peak_memory_mb = metrics.peak_memory_bytes / (1024 * 1024)
            metrics.peak_memory_gb = metrics.peak_memory_mb / 1024

        # 内存分布
        if "model_memory_mb" in data:
            metrics.model_memory_mb = float(data["model_memory_mb"])
        if "optimizer_memory_mb" in data:
            metrics.optimizer_memory_mb = float(data["optimizer_memory_mb"])
        if "activation_memory_mb" in data:
            metrics.activation_memory_mb = float(data["activation_memory_mb"])
        if "gradient_memory_mb" in data:
            metrics.gradient_memory_mb = float(data["gradient_memory_mb"])

        # 碎片率
        if "fragmentation_ratio" in data:
            metrics.fragmentation_ratio = float(data["fragmentation_ratio"])

        # 计算利用率和风险
        metrics.calculate_utilization()

        analysis_data.metrics = metrics

        # Top 内存算子
        if "top_memory_operators" in data:
            analysis_data.top_memory_operators = data["top_memory_operators"]

        # 内存事件
        if "memory_events" in data:
            for event_data in data["memory_events"]:
                if isinstance(event_data, dict):
                    event = MemoryEvent(
                        timestamp=event_data.get("timestamp", 0),
                        operation=event_data.get("operation", ""),
                        size_bytes=event_data.get("size_bytes", 0),
                        total_allocated=event_data.get("total_allocated", 0),
                        operator_name=event_data.get("operator_name", ""),
                    )
                    analysis_data.memory_events.append(event)

        if "memory_timeline" in data and isinstance(data["memory_timeline"], list):
            analysis_data.memory_timeline = data["memory_timeline"]

        return analysis_data

    def _run_core_algorithms(self, analysis_data: MemoryAnalysisData, data: Dict[str, Any]):
        """运行核心分析算法"""

        # 1. 内存泄漏检测
        memory_timeline = data.get("memory_timeline")
        if memory_timeline and len(memory_timeline) > 10:
            leak_result = self.analyzer.detect_leak(memory_timeline)
            analysis_data.leak_result = leak_result

            # 更新指标
            if analysis_data.metrics:
                analysis_data.metrics.has_leak = leak_result.has_leak
                analysis_data.metrics.leak_rate_mb_per_step = leak_result.leak_rate_mb_per_step
                analysis_data.metrics.leak_confidence = leak_result.confidence

        # 2. 内存碎片化检测
        free_blocks = data.get("free_blocks")
        if free_blocks:
            frag_result = self.analyzer.analyze_fragmentation(
                free_blocks,
                analysis_data.metrics.peak_memory_bytes if analysis_data.metrics else 0
            )
            analysis_data.fragmentation_result = frag_result

            # 更新指标
            if analysis_data.metrics:
                analysis_data.metrics.fragmentation_ratio = frag_result.fragmentation_ratio
                analysis_data.metrics.fragmentation_severity = frag_result.severity
                analysis_data.metrics.free_holes = frag_result.free_holes
                analysis_data.metrics.avg_free_block_kb = frag_result.avg_free_block_kb

        # 3. 分配模式分析
        allocations = data.get("allocations")
        if allocations:
            pattern_result = self.analyzer.analyze_allocation_patterns(allocations)
            analysis_data.allocation_pattern = pattern_result

            # 更新指标
            if analysis_data.metrics:
                analysis_data.metrics.avg_allocation_kb = pattern_result.avg_allocation_kb
                analysis_data.metrics.allocation_count = pattern_result.allocation_count
                analysis_data.metrics.small_allocation_ratio = pattern_result.small_allocation_ratio
                analysis_data.metrics.large_allocation_ratio = pattern_result.large_allocation_ratio

    def _extract_recommendations(
        self,
        response: str,
        analysis_data: MemoryAnalysisData
    ) -> List[str]:
        """从 LLM 响应和数据中提取优化建议"""
        recommendations = []

        # 从 LLM 响应提取
        lines = response.split("\n")
        in_recommendation = False

        for line in lines:
            line_lower = line.lower()
            if "建议" in line or "优化" in line or "suggestion" in line_lower:
                in_recommendation = True
            if in_recommendation and line.strip().startswith(("-", "*", "•", "1", "2", "3", "4", "5")):
                clean_line = line.strip().lstrip("-*•0123456789. )")
                if clean_line and len(clean_line) > 5:
                    recommendations.append(clean_line)

        # 基于数据分析添加建议
        metrics = analysis_data.metrics
        if not metrics:
            return recommendations[:10]

        # 内存泄漏建议
        if analysis_data.leak_result and analysis_data.leak_result.has_leak:
            recommendations.insert(0, "⚠️ 检测到内存泄漏，建议检查：")
            recommendations.insert(1, "   - 循环引用是否正确处理")
            recommendations.insert(2, "   - 大对象是否及时释放")
            recommendations.insert(3, "   - 缓存是否有大小限制")

        # 碎片化建议
        if metrics.fragmentation_ratio > 20:
            if len(recommendations) < 5:
                recommendations.append("建议使用内存池预分配减少碎片")
            if metrics.small_allocation_ratio > 0.3:
                recommendations.append("过多的微小分配，考虑使用内存池")

        # OOM 风险建议
        if metrics.oom_risk == "high":
            if metrics.activation_memory_mb > metrics.peak_memory_mb * 0.5:
                recommendations.append("激活值占用过高，建议使用激活值重计算")
            if metrics.model_memory_mb + metrics.optimizer_memory_mb > metrics.peak_memory_mb * 0.6:
                recommendations.append("考虑使用 ZeRO 优化或优化器状态卸载")

        return recommendations[:15]
