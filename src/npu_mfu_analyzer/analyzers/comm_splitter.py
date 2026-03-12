"""
通信拆分模块

按并行策略（TP/DP/PP/CP/EP）拆分通信时间。
复用 msprof-analyze/msprof_analyze/cluster_analyse/cluster_utils/parallel_algorithm.py 的逻辑。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """并行配置"""
    world_size: int = 1
    tensor_parallel_size: int = 1   # TP
    pipeline_parallel_size: int = 1  # PP
    data_parallel_size: int = 1      # DP
    context_parallel_size: int = 1   # CP（序列并行）
    expert_parallel_size: int = 1    # EP（MoE 专家并行）
    
    def __post_init__(self):
        # 验证 world_size = TP * PP * DP
        expected = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        if self.world_size != expected and self.world_size > 1:
            logger.warning(
                f"world_size ({self.world_size}) != TP*PP*DP ({expected}), "
                "may include CP or EP"
            )
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ParallelConfig":
        """从字典创建"""
        return cls(
            world_size=config.get("world_size", 1),
            tensor_parallel_size=config.get("tensor_parallel_size", config.get("tp_size", 1)),
            pipeline_parallel_size=config.get("pipeline_parallel_size", config.get("pp_size", 1)),
            data_parallel_size=config.get("data_parallel_size", config.get("dp_size", 1)),
            context_parallel_size=config.get("context_parallel_size", config.get("cp_size", 1)),
            expert_parallel_size=config.get("expert_parallel_size", config.get("ep_size", 1)),
        )


@dataclass
class CommSplitResult:
    """通信拆分结果"""
    # 各策略的通信时间（单位 us）
    tp_comm_time: float = 0.0
    pp_comm_time: float = 0.0
    dp_comm_time: float = 0.0
    cp_comm_time: float = 0.0
    ep_comm_time: float = 0.0
    other_comm_time: float = 0.0
    total_comm_time: float = 0.0
    
    # 各策略的通信事件数
    tp_comm_count: int = 0
    pp_comm_count: int = 0
    dp_comm_count: int = 0
    cp_comm_count: int = 0
    ep_comm_count: int = 0
    other_comm_count: int = 0
    
    # 各策略的未掩盖通信时间
    tp_not_overlapped: float = 0.0
    pp_not_overlapped: float = 0.0
    dp_not_overlapped: float = 0.0
    
    # 各策略的典型算子
    tp_ops: List[str] = field(default_factory=list)  # AllReduce/AllGather for TP
    pp_ops: List[str] = field(default_factory=list)  # Send/Recv for PP
    dp_ops: List[str] = field(default_factory=list)  # AllReduce for DP
    
    def compute_ratios(self) -> Dict[str, float]:
        """计算各策略占比"""
        total = self.total_comm_time
        if total == 0:
            return {"tp": 0, "pp": 0, "dp": 0, "cp": 0, "ep": 0, "other": 0}
        return {
            "tp": self.tp_comm_time / total * 100,
            "pp": self.pp_comm_time / total * 100,
            "dp": self.dp_comm_time / total * 100,
            "cp": self.cp_comm_time / total * 100,
            "ep": self.ep_comm_time / total * 100,
            "other": self.other_comm_time / total * 100,
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        ratios = self.compute_ratios()
        
        lines = [
            "## 通信拆分分析",
            "",
            "### 各并行策略的通信时间",
            f"| 策略 | 通信时间 | 占比 | 事件数 | 典型算子 |",
            f"|------|---------|------|--------|---------|",
        ]
        
        if self.tp_comm_time > 0:
            lines.append(
                f"| TP (Tensor Parallel) | {self.tp_comm_time/1000:.2f}ms | "
                f"{ratios['tp']:.1f}% | {self.tp_comm_count} | "
                f"{', '.join(self.tp_ops[:3]) or 'AllReduce/AllGather'} |"
            )
        
        if self.pp_comm_time > 0:
            lines.append(
                f"| PP (Pipeline Parallel) | {self.pp_comm_time/1000:.2f}ms | "
                f"{ratios['pp']:.1f}% | {self.pp_comm_count} | "
                f"{', '.join(self.pp_ops[:3]) or 'Send/Recv'} |"
            )
        
        if self.dp_comm_time > 0:
            lines.append(
                f"| DP (Data Parallel) | {self.dp_comm_time/1000:.2f}ms | "
                f"{ratios['dp']:.1f}% | {self.dp_comm_count} | "
                f"{', '.join(self.dp_ops[:3]) or 'AllReduce'} |"
            )
        
        if self.cp_comm_time > 0:
            lines.append(
                f"| CP (Context Parallel) | {self.cp_comm_time/1000:.2f}ms | "
                f"{ratios['cp']:.1f}% | {self.cp_comm_count} | AllGather/ReduceScatter |"
            )
        
        if self.ep_comm_time > 0:
            lines.append(
                f"| EP (Expert Parallel) | {self.ep_comm_time/1000:.2f}ms | "
                f"{ratios['ep']:.1f}% | {self.ep_comm_count} | All2All |"
            )
        
        if self.other_comm_time > 0:
            lines.append(
                f"| Other | {self.other_comm_time/1000:.2f}ms | "
                f"{ratios['other']:.1f}% | {self.other_comm_count} | - |"
            )
        
        lines.append("")
        lines.append(f"**总通信时间**: {self.total_comm_time/1000:.2f} ms")
        
        return "\n".join(lines)


class CommunicationSplitter:
    """
    通信拆分器
    
    将通信事件按并行策略（TP/DP/PP/CP/EP）分类。
    
    识别方法：
    1. 通过通信域名称：如 "tp", "dp", "pp" 前缀
    2. 通过算子类型：
       - P2P (Send/Recv) → PP
       - AllReduce → TP 或 DP
       - All2All → EP (MoE)
       - ReduceScatter/AllGather 在 attention 上下文 → CP
    3. 通过 rank 分组：判断参与通信的 rank 属于哪个域
    """
    
    def __init__(self, parallel_config: Optional[ParallelConfig] = None):
        """
        Args:
            parallel_config: 并行配置，用于辅助判断通信域
        """
        self.config = parallel_config or ParallelConfig()
        
        # 算子类型到策略的映射
        self._op_strategy_map = {
            # PP 相关（点对点通信）
            "send": "pp",
            "isend": "pp",
            "recv": "pp",
            "irecv": "pp",
            "receive": "pp",
            # EP 相关（MoE 专家并行）
            "all2all": "ep",
            "alltoall": "ep",
            "all_to_all": "ep",
        }
    
    def split_from_events(
        self, 
        comm_events: List[Dict[str, Any]],
    ) -> CommSplitResult:
        """
        从通信事件列表拆分
        
        Args:
            comm_events: 通信事件列表，每个事件需包含：
                - name: 算子名称
                - dur: 持续时间
                - args: 可选的附加信息（如 group_name）
                
        Returns:
            CommSplitResult
        """
        result = CommSplitResult()
        
        for event in comm_events:
            name = event.get("name", "").lower()
            dur = event.get("dur", 0)
            args = event.get("args", {})
            group_name = str(args.get("group_name", args.get("groupName", ""))).lower()
            
            strategy = self._identify_strategy(name, group_name)
            
            if strategy == "tp":
                result.tp_comm_time += dur
                result.tp_comm_count += 1
                self._add_unique_op(result.tp_ops, name)
            elif strategy == "pp":
                result.pp_comm_time += dur
                result.pp_comm_count += 1
                self._add_unique_op(result.pp_ops, name)
            elif strategy == "dp":
                result.dp_comm_time += dur
                result.dp_comm_count += 1
                self._add_unique_op(result.dp_ops, name)
            elif strategy == "cp":
                result.cp_comm_time += dur
                result.cp_comm_count += 1
            elif strategy == "ep":
                result.ep_comm_time += dur
                result.ep_comm_count += 1
            else:
                result.other_comm_time += dur
                result.other_comm_count += 1
        
        result.total_comm_time = (
            result.tp_comm_time + result.pp_comm_time + result.dp_comm_time +
            result.cp_comm_time + result.ep_comm_time + result.other_comm_time
        )
        
        return result
    
    def split_from_dataframe(
        self, 
        comm_df: pd.DataFrame,
        name_col: str = "opName",
        dur_col: str = "communication_time",
        group_col: str = "groupName",
    ) -> CommSplitResult:
        """
        从 DataFrame 拆分
        
        Args:
            comm_df: 通信数据 DataFrame
            name_col: 算子名称列
            dur_col: 耗时列
            group_col: 通信域列
            
        Returns:
            CommSplitResult
        """
        result = CommSplitResult()
        
        if comm_df.empty:
            return result
        
        for _, row in comm_df.iterrows():
            name = str(row.get(name_col, "")).lower()
            dur = float(row.get(dur_col, 0))
            group_name = str(row.get(group_col, "")).lower()
            
            strategy = self._identify_strategy(name, group_name)
            
            if strategy == "tp":
                result.tp_comm_time += dur
                result.tp_comm_count += 1
                self._add_unique_op(result.tp_ops, name)
            elif strategy == "pp":
                result.pp_comm_time += dur
                result.pp_comm_count += 1
                self._add_unique_op(result.pp_ops, name)
            elif strategy == "dp":
                result.dp_comm_time += dur
                result.dp_comm_count += 1
                self._add_unique_op(result.dp_ops, name)
            elif strategy == "cp":
                result.cp_comm_time += dur
                result.cp_comm_count += 1
            elif strategy == "ep":
                result.ep_comm_time += dur
                result.ep_comm_count += 1
            else:
                result.other_comm_time += dur
                result.other_comm_count += 1
        
        result.total_comm_time = (
            result.tp_comm_time + result.pp_comm_time + result.dp_comm_time +
            result.cp_comm_time + result.ep_comm_time + result.other_comm_time
        )
        
        return result
    
    def _identify_strategy(self, op_name: str, group_name: str) -> str:
        """
        识别通信事件属于哪个并行策略
        
        Args:
            op_name: 算子名称（小写）
            group_name: 通信域名称（小写）
            
        Returns:
            策略名称：tp, pp, dp, cp, ep, other
        """
        # 1. 从通信域名称识别
        if "tp" in group_name or "tensor" in group_name:
            return "tp"
        if "pp" in group_name or "pipeline" in group_name:
            return "pp"
        if "dp" in group_name or "data" in group_name:
            return "dp"
        if "cp" in group_name or "context" in group_name or "sequence" in group_name:
            return "cp"
        if "ep" in group_name or "expert" in group_name:
            return "ep"
        
        # 2. 从算子类型识别
        for pattern, strategy in self._op_strategy_map.items():
            if pattern in op_name:
                return strategy
        
        # 3. 启发式规则
        # AllReduce 在 TP>1 时通常是 TP，否则可能是 DP
        if "allreduce" in op_name:
            if self.config.tensor_parallel_size > 1:
                return "tp"
            elif self.config.data_parallel_size > 1:
                return "dp"
            return "tp"  # 默认归类为 TP
        
        # ReduceScatter/AllGather 可能是 TP 或 CP
        if "reducescatter" in op_name or "allgather" in op_name:
            if self.config.context_parallel_size > 1:
                return "cp"
            return "tp"
        
        return "other"
    
    def _add_unique_op(self, op_list: List[str], op_name: str, max_ops: int = 5):
        """添加唯一的算子名称"""
        # 清理算子名称
        clean_name = re.sub(r'_\d+$', '', op_name)  # 移除末尾的数字后缀
        if clean_name not in op_list and len(op_list) < max_ops:
            op_list.append(clean_name)


class ParallelGroupBuilder:
    """
    并行组构建器
    
    根据并行配置计算各 rank 所属的并行组。
    复用 msprof-analyze 的 MegatronAlgorithm 逻辑。
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self._tp_groups: List[Set[int]] = []
        self._pp_groups: List[Set[int]] = []
        self._dp_groups: List[Set[int]] = []
        self._build_groups()
    
    def _build_groups(self):
        """构建并行组"""
        world_size = self.config.world_size
        tp = self.config.tensor_parallel_size
        pp = self.config.pipeline_parallel_size
        dp = self.config.data_parallel_size
        
        if world_size <= 1:
            return
        
        # TP Groups: 连续的 tp 个 rank 组成一个 TP 组
        # PP Groups: 间隔 dp*tp 的 rank 组成一个 PP 组
        # DP Groups: 间隔 tp 的 rank 组成一个 DP 组
        
        # 简化实现：假设 rank 按 (TP, DP, PP) 的顺序排列
        for i in range(0, world_size, tp):
            self._tp_groups.append(set(range(i, min(i + tp, world_size))))
        
        for i in range(tp):
            pp_group = set()
            for j in range(pp):
                rank = i + j * (tp * dp)
                if rank < world_size:
                    pp_group.add(rank)
            if pp_group:
                self._pp_groups.append(pp_group)
        
        for i in range(tp * pp):
            dp_group = set()
            for j in range(dp):
                rank = i + j * tp
                if rank < world_size:
                    dp_group.add(rank)
            if dp_group:
                self._dp_groups.append(dp_group)
    
    def get_tp_group(self, rank: int) -> Optional[Set[int]]:
        """获取 rank 所属的 TP 组"""
        for group in self._tp_groups:
            if rank in group:
                return group
        return None
    
    def get_pp_group(self, rank: int) -> Optional[Set[int]]:
        """获取 rank 所属的 PP 组"""
        for group in self._pp_groups:
            if rank in group:
                return group
        return None
    
    def get_dp_group(self, rank: int) -> Optional[Set[int]]:
        """获取 rank 所属的 DP 组"""
        for group in self._dp_groups:
            if rank in group:
                return group
        return None
    
    def identify_comm_strategy_by_ranks(
        self, 
        participating_ranks: Set[int]
    ) -> str:
        """
        根据参与通信的 rank 集合识别策略
        
        Args:
            participating_ranks: 参与通信的 rank 集合
            
        Returns:
            策略名称：tp, pp, dp, other
        """
        # 检查是否匹配某个 TP 组
        for group in self._tp_groups:
            if participating_ranks == group:
                return "tp"
        
        # 检查是否匹配某个 PP 组
        for group in self._pp_groups:
            if participating_ranks == group:
                return "pp"
        
        # 检查是否匹配某个 DP 组
        for group in self._dp_groups:
            if participating_ranks == group:
                return "dp"
        
        # 检查是否是 PP 组的子集（P2P 通信可能只涉及相邻两个 stage）
        for group in self._pp_groups:
            if len(participating_ranks) == 2 and participating_ranks.issubset(group):
                return "pp"
        
        return "other"
