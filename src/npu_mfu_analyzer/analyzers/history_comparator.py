"""
历史对比分析

对比多次 Profiling 结果，分析性能变化趋势。
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProfilingSnapshot:
    """Profiling 快照"""
    id: str
    name: str
    timestamp: str
    profiling_path: str
    
    # 核心指标
    step_count: int = 0
    avg_step_time_us: float = 0.0
    estimated_mfu: float = 0.0
    
    # 时间分布
    compute_ratio: float = 0.0
    comm_ratio: float = 0.0
    idle_ratio: float = 0.0
    overlap_ratio: float = 0.0
    
    # 其他指标
    peak_memory_gb: float = 0.0
    rank_count: int = 1
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "profiling_path": self.profiling_path,
            "step_count": self.step_count,
            "avg_step_time_us": self.avg_step_time_us,
            "estimated_mfu": self.estimated_mfu,
            "compute_ratio": self.compute_ratio,
            "comm_ratio": self.comm_ratio,
            "idle_ratio": self.idle_ratio,
            "overlap_ratio": self.overlap_ratio,
            "peak_memory_gb": self.peak_memory_gb,
            "rank_count": self.rank_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfilingSnapshot":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            timestamp=data.get("timestamp", ""),
            profiling_path=data.get("profiling_path", ""),
            step_count=data.get("step_count", 0),
            avg_step_time_us=data.get("avg_step_time_us", 0.0),
            estimated_mfu=data.get("estimated_mfu", 0.0),
            compute_ratio=data.get("compute_ratio", 0.0),
            comm_ratio=data.get("comm_ratio", 0.0),
            idle_ratio=data.get("idle_ratio", 0.0),
            overlap_ratio=data.get("overlap_ratio", 0.0),
            peak_memory_gb=data.get("peak_memory_gb", 0.0),
            rank_count=data.get("rank_count", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ComparisonResult:
    """对比结果"""
    baseline: ProfilingSnapshot
    current: ProfilingSnapshot
    
    # 变化量
    step_time_change_pct: float = 0.0  # 正数表示变慢
    mfu_change_pct: float = 0.0  # 正数表示提升
    compute_change_pct: float = 0.0
    comm_change_pct: float = 0.0
    idle_change_pct: float = 0.0
    
    # 总体评价
    is_improved: bool = False
    improvement_summary: str = ""
    
    # 详细变化
    changes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        trend_icon = "📈" if self.is_improved else "📉"
        
        lines = [
            f"# 性能对比报告 {trend_icon}",
            "",
            f"**基准版本**: {self.baseline.name} ({self.baseline.timestamp})",
            f"**当前版本**: {self.current.name} ({self.current.timestamp})",
            "",
            "## 核心指标变化",
            "",
            "| 指标 | 基准值 | 当前值 | 变化 |",
            "|------|--------|--------|------|",
            f"| Step 时间 | {self.baseline.avg_step_time_us/1000:.2f}ms | {self.current.avg_step_time_us/1000:.2f}ms | {self._format_change(self.step_time_change_pct, reverse=True)} |",
            f"| 估算 MFU | {self.baseline.estimated_mfu*100:.1f}% | {self.current.estimated_mfu*100:.1f}% | {self._format_change(self.mfu_change_pct)} |",
            f"| 计算占比 | {self.baseline.compute_ratio*100:.1f}% | {self.current.compute_ratio*100:.1f}% | {self._format_change(self.compute_change_pct)} |",
            f"| 通信占比 | {self.baseline.comm_ratio*100:.1f}% | {self.current.comm_ratio*100:.1f}% | {self._format_change(self.comm_change_pct, reverse=True)} |",
            f"| 空闲占比 | {self.baseline.idle_ratio*100:.1f}% | {self.current.idle_ratio*100:.1f}% | {self._format_change(self.idle_change_pct, reverse=True)} |",
            "",
            "## 总结",
            "",
            self.improvement_summary,
        ]
        
        if self.changes:
            lines.extend([
                "",
                "## 详细变化",
                "",
            ])
            for change in self.changes:
                icon = "✅" if change.get("improved", False) else "⚠️"
                lines.append(f"- {icon} {change.get('description', '')}")
        
        return "\n".join(lines)
    
    def _format_change(self, pct: float, reverse: bool = False) -> str:
        """格式化变化百分比"""
        if abs(pct) < 0.1:
            return "➡️ 无变化"
        
        is_good = pct > 0
        if reverse:
            is_good = pct < 0
        
        icon = "✅" if is_good else "⚠️"
        sign = "+" if pct > 0 else ""
        return f"{icon} {sign}{pct:.1f}%"


class HistoryComparator:
    """
    历史对比分析器
    
    功能：
    1. 保存 Profiling 快照
    2. 对比两次 Profiling 结果
    3. 分析性能变化趋势
    4. 生成对比报告
    """
    
    def __init__(self, history_file: Optional[str] = None):
        """
        Args:
            history_file: 历史记录文件路径
        """
        self.history_file = Path(history_file) if history_file else Path("./profiling_history.json")
        self.snapshots: List[ProfilingSnapshot] = []
        
        # 加载历史记录
        self._load_history()
    
    def _load_history(self):
        """加载历史记录"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.snapshots = [
                        ProfilingSnapshot.from_dict(s) for s in data.get("snapshots", [])
                    ]
                logger.info(f"Loaded {len(self.snapshots)} snapshots from history")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
    
    def _save_history(self):
        """保存历史记录"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump({
                    "snapshots": [s.to_dict() for s in self.snapshots],
                    "updated_at": datetime.now().isoformat(),
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.snapshots)} snapshots to history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def add_snapshot(self, snapshot: ProfilingSnapshot):
        """添加快照"""
        self.snapshots.append(snapshot)
        self._save_history()
    
    def create_snapshot_from_summary(
        self,
        name: str,
        profiling_path: str,
        summary: Any,  # ProfilingSummary
        snapshot_id: Optional[str] = None,
    ) -> ProfilingSnapshot:
        """从 ProfilingSummary 创建快照"""
        import uuid
        
        # 提取指标
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else {}
        
        compute_time = summary_dict.get("avg_compute_time", 0)
        comm_time = summary_dict.get("avg_comm_time", 0)
        free_time = summary_dict.get("avg_free_time", 0)
        total_time = compute_time + comm_time + free_time
        
        snapshot = ProfilingSnapshot(
            id=snapshot_id or str(uuid.uuid4())[:8],
            name=name,
            timestamp=datetime.now().isoformat(),
            profiling_path=profiling_path,
            step_count=summary_dict.get("step_count", 0),
            avg_step_time_us=summary_dict.get("avg_step_time", 0),
            estimated_mfu=compute_time / total_time * 0.8 if total_time > 0 else 0,
            compute_ratio=compute_time / total_time if total_time > 0 else 0,
            comm_ratio=comm_time / total_time if total_time > 0 else 0,
            idle_ratio=free_time / total_time if total_time > 0 else 0,
            overlap_ratio=summary_dict.get("overlap_ratio", 0),
            rank_count=summary_dict.get("rank_count", 1),
        )
        
        self.add_snapshot(snapshot)
        return snapshot
    
    def compare(
        self,
        baseline_id: str,
        current_id: str,
    ) -> ComparisonResult:
        """对比两个快照"""
        baseline = self._get_snapshot(baseline_id)
        current = self._get_snapshot(current_id)
        
        if not baseline or not current:
            raise ValueError(f"Snapshot not found: baseline={baseline_id}, current={current_id}")
        
        return self._compare_snapshots(baseline, current)
    
    def compare_with_latest(
        self,
        current_id: str,
    ) -> Optional[ComparisonResult]:
        """与最新的快照对比（排除 current 本身）"""
        current = self._get_snapshot(current_id)
        if not current:
            raise ValueError(f"Snapshot not found: {current_id}")
        
        # 找到最新的其他快照
        other_snapshots = [s for s in self.snapshots if s.id != current_id]
        if not other_snapshots:
            return None
        
        baseline = other_snapshots[-1]  # 最后一个
        return self._compare_snapshots(baseline, current)

    def compare_profiling_paths(
        self,
        path_a: str,
        path_b: str,
        label_a: str = "基准版本",
        label_b: str = "当前版本",
    ) -> ComparisonResult:
        """
        直接通过 Profiling 路径进行对比

        无需预先保存快照，直接加载两个 Profiling 路径的数据进行对比。

        Args:
            path_a: 基准 Profiling 路径
            path_b: 当前 Profiling 路径
            label_a: 基准版本标签
            label_b: 当前版本标签

        Returns:
            ComparisonResult
        """
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader
        from npu_mfu_analyzer.data_loader.data_summarizer import DataSummarizer

        # 加载并摘要化两个 Profiling 数据
        loader_a = ProfilingLoader(path_a)
        loader_b = ProfilingLoader(path_b)
        summary_a = DataSummarizer(loader_a).summarize()
        summary_b = DataSummarizer(loader_b).summarize()

        # 创建快照
        snapshot_a = self.create_snapshot_from_summary(label_a, path_a, summary_a)
        snapshot_b = self.create_snapshot_from_summary(label_b, path_b, summary_b)

        return self._compare_snapshots(snapshot_a, snapshot_b)
    
    def _get_snapshot(self, snapshot_id: str) -> Optional[ProfilingSnapshot]:
        """获取快照"""
        for s in self.snapshots:
            if s.id == snapshot_id:
                return s
        return None
    
    def _compare_snapshots(
        self,
        baseline: ProfilingSnapshot,
        current: ProfilingSnapshot,
    ) -> ComparisonResult:
        """对比两个快照"""
        
        def calc_change_pct(old: float, new: float) -> float:
            if old == 0:
                return 0
            return (new - old) / old * 100
        
        # 计算变化
        step_time_change = calc_change_pct(baseline.avg_step_time_us, current.avg_step_time_us)
        mfu_change = calc_change_pct(baseline.estimated_mfu, current.estimated_mfu)
        compute_change = calc_change_pct(baseline.compute_ratio, current.compute_ratio)
        comm_change = calc_change_pct(baseline.comm_ratio, current.comm_ratio)
        idle_change = calc_change_pct(baseline.idle_ratio, current.idle_ratio)
        
        # 判断是否改进
        # 好的变化：step_time 减少、MFU 增加、计算占比增加、通信/空闲占比减少
        improvements = [
            step_time_change < -1,  # 变快超过 1%
            mfu_change > 1,  # MFU 提升超过 1%
            compute_change > 1,  # 计算占比提升
        ]
        is_improved = sum(improvements) >= 2
        
        # 生成总结
        changes = []
        
        if step_time_change < -5:
            changes.append({"description": f"Step 时间减少 {abs(step_time_change):.1f}%", "improved": True})
        elif step_time_change > 5:
            changes.append({"description": f"Step 时间增加 {step_time_change:.1f}%", "improved": False})
        
        if mfu_change > 5:
            changes.append({"description": f"MFU 提升 {mfu_change:.1f}%", "improved": True})
        elif mfu_change < -5:
            changes.append({"description": f"MFU 下降 {abs(mfu_change):.1f}%", "improved": False})
        
        if idle_change < -10:
            changes.append({"description": f"空闲时间减少 {abs(idle_change):.1f}%", "improved": True})
        elif idle_change > 10:
            changes.append({"description": f"空闲时间增加 {idle_change:.1f}%", "improved": False})
        
        if comm_change < -10:
            changes.append({"description": f"通信开销减少 {abs(comm_change):.1f}%", "improved": True})
        elif comm_change > 10:
            changes.append({"description": f"通信开销增加 {comm_change:.1f}%", "improved": False})
        
        # 生成总结文本
        if is_improved:
            improvement_summary = f"🎉 **性能有所提升**：相比基准版本，训练效率有改善。"
        elif any(c.get("improved") for c in changes):
            improvement_summary = "⚖️ **性能持平**：部分指标有所改善，部分有所下降。"
        else:
            improvement_summary = "⚠️ **性能有所下降**：建议检查最近的配置变更。"
        
        return ComparisonResult(
            baseline=baseline,
            current=current,
            step_time_change_pct=step_time_change,
            mfu_change_pct=mfu_change,
            compute_change_pct=compute_change,
            comm_change_pct=comm_change,
            idle_change_pct=idle_change,
            is_improved=is_improved,
            improvement_summary=improvement_summary,
            changes=changes,
        )
    
    def get_trend(self, metric: str = "estimated_mfu", limit: int = 10) -> List[Tuple[str, float]]:
        """获取指标趋势"""
        recent = self.snapshots[-limit:] if len(self.snapshots) > limit else self.snapshots
        return [(s.timestamp, getattr(s, metric, 0)) for s in recent]
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有快照"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "timestamp": s.timestamp,
                "mfu": s.estimated_mfu,
                "step_time_ms": s.avg_step_time_us / 1000,
            }
            for s in self.snapshots
        ]
