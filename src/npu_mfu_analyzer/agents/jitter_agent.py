"""
Anti-Jitter Agent - 抖动检测 Agent

专门识别由于网络波动、CPU 调度、内存争用等导致的计算/通信不对齐和性能抖动
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
import math

import numpy as np

from npu_mfu_analyzer.agents.base_agent import BaseAgent, AnalysisResult
from npu_mfu_analyzer.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class JitterType:
    """抖动类型"""
    COMPUTE = "compute"          # 计算抖动
    COMMUNICATION = "communication"  # 通信抖动
    ALIGNMENT = "alignment"      # 对齐抖动
    MEMORY = "memory"            # 内存抖动


@dataclass
class JitterMetrics:
    """抖动指标"""
    # 计算抖动
    compute_jitter_std: float = 0.0  # 标准差 (us)
    compute_jitter_cv: float = 0.0   # 变异系数 (CV)
    compute_outliers: int = 0        # 异常值数量
    
    # 通信抖动
    comm_jitter_std: float = 0.0
    comm_jitter_cv: float = 0.0
    comm_outliers: int = 0
    
    # 对齐抖动
    alignment_skew_max: float = 0.0  # 最大对齐偏差 (us)
    alignment_skew_avg: float = 0.0  # 平均对齐偏差 (us)
    
    # 跨 rank 抖动
    cross_rank_variance: float = 0.0  # 跨 rank 时间方差
    slow_ranks: List[int] = field(default_factory=list)  # 慢 rank 列表
    
    # 根因分析
    root_causes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compute_jitter_std": self.compute_jitter_std,
            "compute_jitter_cv": self.compute_jitter_cv,
            "compute_outliers": self.compute_outliers,
            "comm_jitter_std": self.comm_jitter_std,
            "comm_jitter_cv": self.comm_jitter_cv,
            "comm_outliers": self.comm_outliers,
            "alignment_skew_max": self.alignment_skew_max,
            "alignment_skew_avg": self.alignment_skew_avg,
            "cross_rank_variance": self.cross_rank_variance,
            "slow_ranks": self.slow_ranks,
            "root_causes": self.root_causes,
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 抖动检测结果",
            "",
        ]
        
        # 计算抖动
        if self.compute_jitter_cv > 0.1:  # CV > 10% 认为有抖动
            lines.extend([
                "### 计算抖动",
                f"- **标准差**: {self.compute_jitter_std:.2f} us",
                f"- **变异系数 (CV)**: {self.compute_jitter_cv:.1%}",
                f"- **异常值数量**: {self.compute_outliers}",
                "",
            ])
        
        # 通信抖动
        if self.comm_jitter_cv > 0.1:
            lines.extend([
                "### 通信抖动",
                f"- **标准差**: {self.comm_jitter_std:.2f} us",
                f"- **变异系数 (CV)**: {self.comm_jitter_cv:.1%}",
                f"- **异常值数量**: {self.comm_outliers}",
                "",
            ])
        
        # 对齐抖动
        if self.alignment_skew_max > 1000:  # 超过 1ms
            lines.extend([
                "### 对齐抖动",
                f"- **最大偏差**: {self.alignment_skew_max/1000:.2f} ms",
                f"- **平均偏差**: {self.alignment_skew_avg/1000:.2f} ms",
                "",
            ])
        
        # 慢 rank
        if self.slow_ranks:
            lines.extend([
                "### 慢 Rank",
                f"检测到 {len(self.slow_ranks)} 个慢 rank: {self.slow_ranks[:10]}",
                "",
            ])
        
        # 根因分析
        if self.root_causes:
            lines.extend([
                "### 可能原因",
            ])
            for cause in self.root_causes:
                lines.append(f"- {cause}")
        
        return "\n".join(lines)


class JitterDetector:
    """
    抖动检测器
    
    检测类型：
    1. 计算抖动：同一算子在不同 Step 的时间标准差
    2. 通信抖动：同一集合操作在不同 Rank 的开始时间差
    3. 对齐抖动：Compute-Comm 边界的对齐偏差
    """
    
    # 抖动阈值
    CV_THRESHOLD = 0.15  # 变异系数阈值 15%
    OUTLIER_THRESHOLD = 3.0  # 3-sigma 阈值
    
    def __init__(self):
        pass
    
    def detect_compute_jitter(
        self,
        operator_traces: Dict[str, List[float]],
    ) -> Tuple[float, float, int]:
        """
        检测计算抖动
        
        Args:
            operator_traces: {op_name: [duration1, duration2, ...]}
            
        Returns:
            (std, cv, outlier_count)
        """
        all_cvs = []
        total_outliers = 0
        
        for op_name, durations in operator_traces.items():
            if len(durations) < 5:  # 至少需要 5 个样本
                continue
            
            mean = np.mean(durations)
            std = np.std(durations)
            
            if mean > 0:
                cv = std / mean
                all_cvs.append(cv)
                
                # 检测异常值（3-sigma 规则）
                outliers = sum(
                    1 for d in durations 
                    if abs(d - mean) > self.OUTLIER_THRESHOLD * std
                )
                total_outliers += outliers
        
        if not all_cvs:
            return 0.0, 0.0, 0
        
        # 返回平均 CV
        avg_cv = np.mean(all_cvs)
        # 计算所有数据的总体标准差
        all_durations = []
        for durations in operator_traces.values():
            all_durations.extend(durations)
        overall_std = np.std(all_durations) if all_durations else 0.0
        
        return overall_std, avg_cv, total_outliers
    
    def detect_communication_jitter(
        self,
        comm_events: List[Dict[str, Any]],
    ) -> Tuple[float, float, int]:
        """
        检测通信抖动
        
        Args:
            comm_events: 通信事件列表
            
        Returns:
            (std, cv, outlier_count)
        """
        # 按操作类型分组
        comm_by_type = {}
        
        for event in comm_events:
            op_name = event.get("name", "unknown")
            duration = event.get("dur", event.get("duration", 0))
            
            if duration <= 0:
                continue
            
            if op_name not in comm_by_type:
                comm_by_type[op_name] = []
            comm_by_type[op_name].append(duration)
        
        # 复用计算抖动的逻辑
        return self.detect_compute_jitter(comm_by_type)
    
    def detect_alignment_jitter(
        self,
        compute_events: List[Dict[str, Any]],
        comm_events: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """
        检测对齐抖动
        
        分析计算和通信事件之间的时间间隙，检测是否存在对齐不良
        
        Returns:
            (max_skew, avg_skew)
        """
        skews = []
        
        # 按时间戳排序
        compute_events = sorted(
            compute_events,
            key=lambda x: x.get("ts", x.get("start_time", 0))
        )
        comm_events = sorted(
            comm_events,
            key=lambda x: x.get("ts", x.get("start_time", 0))
        )
        
        # 找到每个计算事件后的第一个通信事件
        for compute_event in compute_events:
            compute_end = (
                compute_event.get("ts", 0) + 
                compute_event.get("dur", compute_event.get("duration", 0))
            )
            
            # 找最近的通信事件
            for comm_event in comm_events:
                comm_start = comm_event.get("ts", comm_event.get("start_time", 0))
                
                if comm_start >= compute_end:
                    # 计算间隙
                    skew = comm_start - compute_end
                    if 0 <= skew <= 100000:  # 合理范围内（< 100ms）
                        skews.append(skew)
                    break
        
        if not skews:
            return 0.0, 0.0
        
        return max(skews), np.mean(skews)
    
    def detect_cross_rank_jitter(
        self,
        rank_durations: Dict[int, List[float]],
    ) -> Tuple[float, List[int]]:
        """
        检测跨 rank 抖动
        
        Args:
            rank_durations: {rank: [step1_duration, step2_duration, ...]}
            
        Returns:
            (variance, slow_ranks)
        """
        if len(rank_durations) < 2:
            return 0.0, []
        
        # 计算每个 step 的跨 rank 方差
        num_steps = min(len(durations) for durations in rank_durations.values())
        
        variances = []
        for step_idx in range(num_steps):
            step_durations = [
                durations[step_idx] 
                for durations in rank_durations.values()
            ]
            variance = np.var(step_durations)
            variances.append(variance)
        
        avg_variance = np.mean(variances) if variances else 0.0
        
        # 识别慢 rank（平均时间超过中位数 20%）
        avg_by_rank = {
            rank: np.mean(durations)
            for rank, durations in rank_durations.items()
        }
        median = np.median(list(avg_by_rank.values()))
        
        slow_ranks = [
            rank for rank, avg in avg_by_rank.items()
            if avg > median * 1.2
        ]
        
        return avg_variance, slow_ranks
    
    def analyze_root_causes(
        self,
        metrics: JitterMetrics,
        host_events: Optional[List[Dict]] = None,
        memory_events: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        根因分析
        
        基于抖动模式推断可能的根本原因
        """
        causes = []
        
        # 计算抖动高 → CPU 调度问题
        if metrics.compute_jitter_cv > 0.2:
            causes.append(
                "计算抖动较大 (CV > 20%)，可能是 CPU 调度延迟或算子性能不稳定"
            )
        
        # 通信抖动高 → 网络问题
        if metrics.comm_jitter_cv > 0.2:
            causes.append(
                "通信抖动较大 (CV > 20%)，可能是网络拥塞或 RDMA 不稳定"
            )
        
        # 对齐抖动大 → 同步问题
        if metrics.alignment_skew_max > 5000:  # 超过 5ms
            causes.append(
                f"对齐偏差较大 ({metrics.alignment_skew_max/1000:.1f}ms)，"
                "可能是计算-通信流水线未优化"
            )
        
        # 跨 rank 方差大 → 负载不均或慢卡
        if metrics.cross_rank_variance > 1000000:  # 方差超过 1ms^2
            causes.append(
                "跨 rank 时间方差大，可能存在负载不均或硬件异常"
            )
        
        # 有慢 rank → 硬件问题
        if metrics.slow_ranks:
            causes.append(
                f"检测到 {len(metrics.slow_ranks)} 个慢 rank，"
                "可能是硬件性能差异或资源争用"
            )
        
        # Host 事件多 → Host 开销大
        if host_events and len(host_events) > 1000:
            causes.append(
                "Host 事件数量较多，可能是 Python 层开销或 Host-Device 同步过多"
            )
        
        # 内存事件异常 → 内存争用
        if memory_events:
            # 简化检查：如果有内存分配/释放事件
            alloc_events = [e for e in memory_events if "alloc" in e.get("name", "").lower()]
            if len(alloc_events) > 100:
                causes.append(
                    "频繁的内存分配/释放，可能导致内存碎片化和性能抖动"
                )
        
        return causes


class JitterAgent(BaseAgent):
    """
    抖动检测 Agent

    专门识别和分析性能抖动问题
    """

    # 添加 Prompt 模板
    PROMPT_TEMPLATE = """你是一个大模型训练性能专家。请分析以下抖动检测结果，给出专业的诊断和优化建议。

## 抖动指标

### 计算抖动
- 标准差: {compute_jitter_std:.2f} us
- 变异系数: {compute_jitter_cv:.2f}
- 异常值数量: {compute_outliers}

### 通信抖动
- 标准差: {comm_jitter_std:.2f} us
- 变异系数: {comm_jitter_cv:.2f}
- 异常值数量: {comm_outliers}

### 对齐抖动
- 最大对齐偏差: {alignment_skew_max:.2f} us
- 平均对齐偏差: {alignment_skew_avg:.2f} us

### 跨 Rank 抖动
- 跨 Rank 时间方差: {cross_rank_variance:.2f}
- 慢 Rank: {slow_ranks}

## 分析要求

请基于以上指标，分析：
1. 抖动的严重程度评估
2. 可能的根本原因
3. 具体的优化建议（优先级排序）
"""

    def __init__(
        self,
        llm: LLMInterface,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="JitterAgent",
            llm=llm,
            system_prompt="你是大模型训练性能分析专家，专注于识别和诊断性能抖动问题。",
            config=config
        )
        self.detector = JitterDetector()

    def get_prompt_template(self) -> str:
        """获取 Prompt 模板"""
        return self.PROMPT_TEMPLATE

    async def analyze(self, profiling_data: Dict[str, Any]) -> AnalysisResult:
        """
        分析抖动

        Args:
            profiling_data: Profiling 数据

        Returns:
            AnalysisResult
        """
        try:
            # 提取数据
            operator_traces = profiling_data.get("operator_traces", {})
            comm_events = profiling_data.get("comm_events", [])
            compute_events = profiling_data.get("compute_events", [])
            rank_durations = profiling_data.get("rank_durations", {})
            host_events = profiling_data.get("host_events", [])
            memory_events = profiling_data.get("memory_events", [])

            # 1. 检测计算抖动
            compute_std, compute_cv, compute_outliers = self.detector.detect_compute_jitter(
                operator_traces
            )

            # 2. 检测通信抖动
            comm_std, comm_cv, comm_outliers = self.detector.detect_communication_jitter(
                comm_events
            )

            # 3. 检测对齐抖动
            max_skew, avg_skew = self.detector.detect_alignment_jitter(
                compute_events, comm_events
            )

            # 4. 检测跨 rank 抖动
            cross_rank_var, slow_ranks = self.detector.detect_cross_rank_jitter(
                rank_durations
            )

            # 5. 构建指标
            metrics = JitterMetrics(
                compute_jitter_std=compute_std,
                compute_jitter_cv=compute_cv,
                compute_outliers=compute_outliers,
                comm_jitter_std=comm_std,
                comm_jitter_cv=comm_cv,
                comm_outliers=comm_outliers,
                alignment_skew_max=max_skew,
                alignment_skew_avg=avg_skew,
                cross_rank_variance=cross_rank_var,
                slow_ranks=slow_ranks,
            )

            # 6. 根因分析
            metrics.root_causes = self.detector.analyze_root_causes(
                metrics, host_events, memory_events
            )

            # 7. 生成 Prompt
            data_summary = self._format_data_summary(metrics)
            prompt = self.format_prompt(self.PROMPT_TEMPLATE, **metrics.to_dict())

            # 8. 调用 LLM
            response = await self.call_llm(prompt)

            return AnalysisResult(
                agent_name=self.name,
                success=True,
                summary="Jitter 分析完成",
                details={"metrics": metrics.to_dict()},
                recommendations=self._extract_recommendations(response),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Jitter analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                success=False,
                summary="Jitter 分析失败",
                error=str(e),
            )

    def _format_data_summary(self, metrics: JitterMetrics) -> str:
        """格式化数据摘要"""
        lines = [
            f"- 计算抖动标准差: {metrics.compute_jitter_std:.2f} us",
            f"- 计算抖动变异系数: {metrics.compute_jitter_cv:.2f}",
            f"- 计算异常值数量: {metrics.compute_outliers}",
            f"- 通信抖动标准差: {metrics.comm_jitter_std:.2f} us",
            f"- 通信抖动变异系数: {metrics.comm_jitter_cv:.2f}",
            f"- 通信异常值数量: {metrics.comm_outliers}",
            f"- 最大对齐偏差: {metrics.alignment_skew_max:.2f} us",
            f"- 平均对齐偏差: {metrics.alignment_skew_avg:.2f} us",
            f"- 跨 Rank 时间方差: {metrics.cross_rank_variance:.2f}",
        ]
        if metrics.slow_ranks:
            lines.append(f"- 慢 Rank: {metrics.slow_ranks}")
        return "\n".join(lines)


def detect_jitter_from_loader(loader) -> JitterMetrics:
    """
    从 ProfilingLoader 检测抖动
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        JitterMetrics
    """
    detector = JitterDetector()
    
    # 收集数据
    operator_traces = {}
    comm_events = []
    compute_events = []
    
    try:
        overlap_events = loader.get_overlap_events()
        comm_events = overlap_events.get("hccl", [])
        compute_events = overlap_events.get("compute", [])
    except Exception as e:
        logger.warning(f"Failed to collect events: {e}")
    
    # 检测各类抖动
    compute_std, compute_cv, compute_outliers = detector.detect_compute_jitter(
        operator_traces
    )
    comm_std, comm_cv, comm_outliers = detector.detect_communication_jitter(
        comm_events
    )
    max_skew, avg_skew = detector.detect_alignment_jitter(
        compute_events, comm_events
    )
    
    metrics = JitterMetrics(
        compute_jitter_std=compute_std,
        compute_jitter_cv=compute_cv,
        compute_outliers=compute_outliers,
        comm_jitter_std=comm_std,
        comm_jitter_cv=comm_cv,
        comm_outliers=comm_outliers,
        alignment_skew_max=max_skew,
        alignment_skew_avg=avg_skew,
    )
    
    metrics.root_causes = detector.analyze_root_causes(metrics)
    
    return metrics
