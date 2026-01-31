"""
慢卡检测模块

检测集群训练中的慢卡（性能异常的 Rank），使用 Dixon 检验和三倍标准差法。
复用自 msprof-analyze/msprof_analyze/cluster_analyse/recipes/slow_rank/slow_rank.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Dixon 检验临界值表（99.5% 置信度）
# 复用自 msprof-analyze/msprof_analyze/cluster_analyse/recipes/slow_rank/dixon_table.py
DIXON_TABLE_995 = {
    3: 0.994, 4: 0.920, 5: 0.823, 6: 0.744, 7: 0.680,
    8: 0.723, 9: 0.676, 10: 0.638, 11: 0.707, 12: 0.675,
    13: 0.649, 14: 0.672, 15: 0.649, 16: 0.629, 17: 0.611,
    18: 0.595, 19: 0.580, 20: 0.568, 21: 0.556, 22: 0.545,
    23: 0.536, 24: 0.526, 25: 0.519, 26: 0.510, 27: 0.503,
    28: 0.496, 29: 0.489, 30: 0.484, 31: 0.478, 32: 0.473,
    33: 0.468, 34: 0.463, 35: 0.458, 36: 0.454, 37: 0.450,
    38: 0.446, 39: 0.442, 40: 0.439, 41: 0.435, 42: 0.432,
    43: 0.429, 44: 0.425, 45: 0.423, 46: 0.420, 47: 0.417,
    48: 0.414, 49: 0.412, 50: 0.409, 51: 0.407, 52: 0.405,
    53: 0.402, 54: 0.400, 55: 0.398, 56: 0.396, 57: 0.394,
    58: 0.392, 59: 0.391, 60: 0.388, 61: 0.387, 62: 0.385,
    63: 0.383, 64: 0.382, 65: 0.380, 66: 0.379, 67: 0.377,
    68: 0.376, 69: 0.374, 70: 0.372, 71: 0.371, 72: 0.370,
    73: 0.368, 74: 0.368, 75: 0.366, 76: 0.365, 77: 0.364,
    78: 0.363, 79: 0.361, 80: 0.360, 81: 0.359, 82: 0.358,
    83: 0.356, 84: 0.356, 85: 0.355, 86: 0.353, 87: 0.352,
    88: 0.352, 89: 0.351, 90: 0.350, 91: 0.349, 92: 0.348,
    93: 0.347, 94: 0.346, 95: 0.345, 96: 0.344, 97: 0.344,
    98: 0.343, 99: 0.341, 100: 0.341,
}

# Dixon 检验阈值常量
MAX_DIXON_NUM = 100  # Dixon 检验最大样本数
DIXON_THRESHOLD_1 = 7   # 使用第一种公式的阈值
DIXON_THRESHOLD_2 = 10  # 使用第二种公式的阈值
DIXON_THRESHOLD_3 = 13  # 使用第三种公式的阈值


def judge_norm(time_list: List[float], threshold: int = 3) -> List[int]:
    """
    三倍标准差法检测异常值
    
    适用于大样本（>25 个 Rank）
    
    慢卡判定逻辑：
    1. 耗时 < mean - 3*std 的卡（通信场景下耗时短说明在等待）
    2. 如果存在耗时 > mean + 3*std 的卡，则最短耗时的卡也是慢卡
    
    Args:
        time_list: 各 Rank 的耗时列表
        threshold: 标准差倍数（默认 3）
        
    Returns:
        异常值的索引列表
    """
    if len(time_list) < 3:
        return []
    
    t_max = max(time_list)
    t_min = min(time_list)
    t_mean = np.mean(time_list)
    t_std = np.std(time_list)
    
    if t_std == 0:
        return []  # 所有值相同，无异常
    
    threshold_high = t_mean + threshold * t_std
    threshold_low = t_mean - threshold * t_std
    
    # 耗时低于下阈值的卡认为是慢卡（通信场景下耗时短说明在等待其他卡）
    outliers_idx = [i for i, t in enumerate(time_list) if t < threshold_low]
    
    # 如果存在高于上阈值的卡，则将耗时最短的卡加到慢卡 list 中
    if t_max > threshold_high:
        min_idx = time_list.index(t_min)
        if min_idx not in outliers_idx:
            outliers_idx.append(min_idx)
    
    return outliers_idx


def judge_dixon(time_list: List[float]) -> List[int]:
    """
    Dixon 检验法检测异常值
    
    适用于小样本（<=25 个 Rank）
    
    检验逻辑：
    1. 计算检验指标 = (次小值 - 最小值) / (最大值 - 最小值)
    2. 根据样本量查表，若指标 > 临界值，则最小值为异常值（慢卡）
    
    Args:
        time_list: 各 Rank 的耗时列表
        
    Returns:
        异常值的索引列表
    """
    n = len(time_list)
    if n <= 2:
        return []
    
    sorted_list = sorted(time_list)
    
    # 判断分母是否可能为 0（值过于集中）
    if len(set(sorted_list)) <= 3:
        return []
    
    # 根据样本量选择不同的计算公式
    if n > MAX_DIXON_NUM:
        return []  # 样本过大，应使用三倍标准差
    
    if n <= DIXON_THRESHOLD_1:
        # 小样本：flag = (x2 - x1) / (xn - x1)
        denominator = sorted_list[-1] - sorted_list[0]
        flag = (sorted_list[1] - sorted_list[0]) / denominator if denominator else 0
    elif n <= DIXON_THRESHOLD_2:
        # 中样本：flag = (x2 - x1) / (x(n-1) - x1)
        denominator = sorted_list[-2] - sorted_list[0]
        flag = (sorted_list[1] - sorted_list[0]) / denominator if denominator else 0
    elif n <= DIXON_THRESHOLD_3:
        # 较大样本：flag = (x3 - x1) / (x(n-1) - x1)
        denominator = sorted_list[-2] - sorted_list[0]
        flag = (sorted_list[2] - sorted_list[0]) / denominator if denominator else 0
    else:
        # 大样本：flag = (x3 - x1) / (x(n-2) - x1)
        denominator = sorted_list[-3] - sorted_list[0]
        flag = (sorted_list[2] - sorted_list[0]) / denominator if denominator else 0
    
    # 查表判断是否异常
    critical_value = DIXON_TABLE_995.get(n, 0)
    if flag > critical_value:
        return [time_list.index(sorted_list[0])]
    
    return []


def judge_slow_rank(time_list: List[float]) -> List[int]:
    """
    根据样本量自动选择检验方法检测慢卡
    
    - 样本 <= 25：使用 Dixon 检验
    - 样本 > 25：使用三倍标准差
    
    Args:
        time_list: 各 Rank 的耗时列表
        
    Returns:
        慢卡的索引列表
    """
    if len(time_list) <= MAX_DIXON_NUM:
        return judge_dixon(time_list)
    else:
        return judge_norm(time_list)


@dataclass
class SlowRankInfo:
    """慢卡信息"""
    rank_id: int
    dimension: str  # "computing", "free", "communication"
    value: float    # 该维度的耗时
    mean_value: float  # 平均值
    std_value: float   # 标准差
    deviation: float   # 偏离程度（与平均值的差异）


@dataclass
class SlowRankResult:
    """慢卡检测结果"""
    slow_ranks_by_compute: List[int] = field(default_factory=list)   # 计算慢卡
    slow_ranks_by_free: List[int] = field(default_factory=list)      # 等待慢卡
    slow_ranks_by_comm: List[int] = field(default_factory=list)      # 通信慢卡
    slow_rank_details: List[SlowRankInfo] = field(default_factory=list)  # 详细信息
    bottleneck_ops: List[Dict[str, Any]] = field(default_factory=list)   # 瓶颈通信算子
    
    def has_slow_ranks(self) -> bool:
        """是否检测到慢卡"""
        return bool(
            self.slow_ranks_by_compute or 
            self.slow_ranks_by_free or 
            self.slow_ranks_by_comm
        )
    
    def get_all_slow_ranks(self) -> List[int]:
        """获取所有慢卡（去重）"""
        all_ranks = set(self.slow_ranks_by_compute + self.slow_ranks_by_free + self.slow_ranks_by_comm)
        return sorted(list(all_ranks))
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        if not self.has_slow_ranks():
            return "## 慢卡检测\n未检测到明显的慢卡问题。"
        
        lines = ["## 慢卡检测结果"]
        
        if self.slow_ranks_by_compute:
            lines.append(f"- **计算慢卡**（负载重/硬件慢）: Rank {self.slow_ranks_by_compute}")
        if self.slow_ranks_by_free:
            lines.append(f"- **等待慢卡**（被其他卡拖慢）: Rank {self.slow_ranks_by_free}")
        if self.slow_ranks_by_comm:
            lines.append(f"- **通信慢卡**（链路/拓扑问题）: Rank {self.slow_ranks_by_comm}")
        
        if self.slow_rank_details:
            lines.append("\n### 慢卡详情")
            for info in self.slow_rank_details[:5]:  # 只显示前 5 个
                deviation_pct = (info.deviation / info.mean_value * 100) if info.mean_value else 0
                lines.append(
                    f"- Rank {info.rank_id} [{info.dimension}]: "
                    f"{info.value:.2f} us (平均 {info.mean_value:.2f}, 偏离 {deviation_pct:.1f}%)"
                )
        
        return "\n".join(lines)


class SlowRankDetector:
    """
    慢卡检测器
    
    多维度检测集群训练中的慢卡：
    1. 基于计算时间（Computing）：计算慢说明负载重或硬件慢
    2. 基于空闲时间（Free）：空闲多说明在等待其他卡
    3. 基于通信时间（Communication）：通信慢说明链路或拓扑问题
    """
    
    def __init__(self):
        pass
    
    def detect_from_step_trace(
        self, 
        step_trace_df: pd.DataFrame,
        rank_column: str = "rank",
    ) -> SlowRankResult:
        """
        从 STEP_TRACE 数据检测慢卡
        
        Args:
            step_trace_df: 包含各 Rank 耗时数据的 DataFrame
                          需要有 rank, computing, communication, free 等列
            rank_column: Rank ID 列名
            
        Returns:
            SlowRankResult: 慢卡检测结果
        """
        result = SlowRankResult()
        
        if step_trace_df.empty:
            return result
        
        # 尝试查找 rank 列
        rank_col = None
        for col in [rank_column, "rank", "Rank", "rank_id", "rankId", "device_id"]:
            if col in step_trace_df.columns:
                rank_col = col
                break
        
        # 如果没有 rank 列，尝试使用 index 或 type
        if rank_col is None:
            if "type" in step_trace_df.columns and "index" in step_trace_df.columns:
                # 过滤 type='rank' 的数据
                rank_df = step_trace_df[step_trace_df["type"] == "rank"]
                if not rank_df.empty:
                    step_trace_df = rank_df
                    rank_col = "index"
        
        if rank_col is None:
            logger.warning("Cannot find rank column in step_trace_df")
            return result
        
        # 按 rank 分组计算平均值
        grouped = step_trace_df.groupby(rank_col)
        
        # 1. 基于计算时间检测
        if "computing" in step_trace_df.columns:
            compute_times = grouped["computing"].mean()
            time_list = compute_times.tolist()
            slow_indices = judge_slow_rank(time_list)
            result.slow_ranks_by_compute = [compute_times.index[i] for i in slow_indices]
            
            # 记录详情
            for idx in slow_indices:
                result.slow_rank_details.append(SlowRankInfo(
                    rank_id=compute_times.index[idx],
                    dimension="computing",
                    value=time_list[idx],
                    mean_value=np.mean(time_list),
                    std_value=np.std(time_list),
                    deviation=abs(time_list[idx] - np.mean(time_list)),
                ))
        
        # 2. 基于空闲时间检测
        if "free" in step_trace_df.columns:
            free_times = grouped["free"].mean()
            time_list = free_times.tolist()
            # 注意：空闲时间多的卡是慢卡（在等待），所以用 max 检测
            slow_indices = self._detect_high_outliers(time_list)
            result.slow_ranks_by_free = [free_times.index[i] for i in slow_indices]
            
            for idx in slow_indices:
                result.slow_rank_details.append(SlowRankInfo(
                    rank_id=free_times.index[idx],
                    dimension="free",
                    value=time_list[idx],
                    mean_value=np.mean(time_list),
                    std_value=np.std(time_list),
                    deviation=abs(time_list[idx] - np.mean(time_list)),
                ))
        
        # 3. 基于通信时间检测
        if "communication" in step_trace_df.columns:
            comm_times = grouped["communication"].mean()
            time_list = comm_times.tolist()
            slow_indices = judge_slow_rank(time_list)
            result.slow_ranks_by_comm = [comm_times.index[i] for i in slow_indices]
            
            for idx in slow_indices:
                result.slow_rank_details.append(SlowRankInfo(
                    rank_id=comm_times.index[idx],
                    dimension="communication",
                    value=time_list[idx],
                    mean_value=np.mean(time_list),
                    std_value=np.std(time_list),
                    deviation=abs(time_list[idx] - np.mean(time_list)),
                ))
        
        return result
    
    def _detect_high_outliers(self, time_list: List[float], threshold: int = 3) -> List[int]:
        """
        检测高异常值（空闲时间多的慢卡）
        
        与 judge_norm 相反，这里检测的是耗时 > mean + 3*std 的卡
        """
        if len(time_list) < 3:
            return []
        
        t_mean = np.mean(time_list)
        t_std = np.std(time_list)
        
        if t_std == 0:
            return []
        
        threshold_high = t_mean + threshold * t_std
        return [i for i, t in enumerate(time_list) if t > threshold_high]
    
    def detect_bottleneck_ops(
        self, 
        comm_ops_df: pd.DataFrame,
        group_col: str = "groupName",
        op_col: str = "opName",
        rank_col: str = "rankId",
        time_col: str = "communication_time",
    ) -> List[Dict[str, Any]]:
        """
        检测瓶颈通信算子
        
        分析每个通信算子在各 Rank 上的耗时差异，找出存在慢卡的算子
        
        Args:
            comm_ops_df: 通信算子数据
            group_col: 通信域列名
            op_col: 算子名称列名
            rank_col: Rank ID 列名
            time_col: 耗时列名
            
        Returns:
            瓶颈算子列表
        """
        bottleneck_ops = []
        
        if comm_ops_df.empty:
            return bottleneck_ops
        
        # 按通信域和算子名分组
        for (group_name, op_name), group_df in comm_ops_df.groupby([group_col, op_col]):
            time_list = group_df[time_col].tolist()
            slow_indices = judge_slow_rank(time_list)
            
            if slow_indices:
                slow_ranks = [group_df.iloc[i][rank_col] for i in slow_indices]
                bottleneck_ops.append({
                    "group_name": group_name,
                    "op_name": op_name,
                    "slow_ranks": slow_ranks,
                    "mean_time": np.mean(time_list),
                    "std_time": np.std(time_list),
                    "min_time": min(time_list),
                    "max_time": max(time_list),
                })
        
        # 按影响程度排序（标准差越大影响越大）
        bottleneck_ops.sort(key=lambda x: x["std_time"], reverse=True)
        
        return bottleneck_ops
