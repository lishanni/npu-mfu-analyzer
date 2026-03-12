"""
并行模式检测器

自动识别并行策略配置（TP、PP、DP、ZeRO、FSDP 等）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """并行策略类型"""
    TENSOR_PARALLEL = "tp"
    PIPELINE_PARALLEL = "pp"
    DATA_PARALLEL = "dp"
    ZERO_STAGE_1 = "zero1"
    ZERO_STAGE_2 = "zero2"
    ZERO_STAGE_3 = "zero3"
    FSDP = "fsdp"
    CONTEXT_PARALLEL = "cp"
    EXPERT_PARALLEL = "ep"


@dataclass
class ParallelConfig:
    """并行配置"""
    world_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    
    # 高级策略
    use_zero: bool = False
    zero_stage: Optional[int] = None
    use_fsdp: bool = False
    
    # 检测置信度
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """验证配置合理性"""
        if self.world_size > 1:
            expected = (
                self.tensor_parallel_size * 
                self.pipeline_parallel_size * 
                self.data_parallel_size
            )
            if self.context_parallel_size > 1:
                expected *= self.context_parallel_size
            if self.expert_parallel_size > 1:
                expected *= self.expert_parallel_size
            
            if expected != self.world_size and expected > 1:
                logger.warning(
                    f"Parallel config inconsistent: world_size={self.world_size}, "
                    f"TP*PP*DP*CP*EP={expected}"
                )


class ParallelDetector:
    """
    并行模式检测器
    
    通过分析通信模式、算子特征、Rank 分组等信息，
    自动推断出并行策略配置。
    """
    
    def __init__(self, world_size: int):
        """
        Args:
            world_size: 总进程数
        """
        self.world_size = world_size
    
    def detect(
        self,
        comm_events: List[Dict[str, Any]],
        operator_names: Optional[List[str]] = None,
    ) -> ParallelConfig:
        """
        检测并行配置
        
        Args:
            comm_events: 通信事件列表
            operator_names: 算子名称列表
            
        Returns:
            ParallelConfig
        """
        if operator_names is None:
            operator_names = []
        
        evidence = []
        
        # 1. 从通信组分析并行度
        tp_size = self._detect_tp_size(comm_events, evidence)
        pp_size = self._detect_pp_size(comm_events, evidence)
        dp_size = self._detect_dp_size(comm_events, evidence)
        cp_size = self._detect_cp_size(comm_events, evidence)
        ep_size = self._detect_ep_size(comm_events, evidence)
        
        # 2. 检测 ZeRO 和 FSDP
        use_zero, zero_stage = self._detect_zero(comm_events, operator_names, evidence)
        use_fsdp = self._detect_fsdp(comm_events, operator_names, evidence)
        
        # 3. 如果检测不到，尝试推断
        if tp_size == 1 and pp_size == 1 and dp_size == 1:
            tp_size, pp_size, dp_size = self._infer_parallel_sizes(
                self.world_size, comm_events, evidence
            )
        
        # 4. 计算置信度
        confidence = self._calculate_confidence(evidence, comm_events)
        
        return ParallelConfig(
            world_size=self.world_size,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            context_parallel_size=cp_size,
            expert_parallel_size=ep_size,
            use_zero=use_zero,
            zero_stage=zero_stage,
            use_fsdp=use_fsdp,
            confidence=confidence,
            evidence=evidence,
        )
    
    def _detect_tp_size(self, comm_events: List[Dict], evidence: List[str]) -> int:
        """检测 Tensor Parallel 大小"""
        # 查找 TP 相关的 AllReduce/AllGather
        tp_groups = set()
        
        for event in comm_events:
            name = event.get("name", "").lower()
            args = event.get("args", {})
            group_name = str(args.get("group_name", args.get("groupName", ""))).lower()
            
            # TP 特征：
            # 1. 通信组名称包含 "tp" 或 "tensor"
            # 2. AllReduce 操作（梯度聚合）
            if ("tp" in group_name or "tensor" in group_name) and "allreduce" in name:
                # 提取组大小（如果有）
                group_size = args.get("group_size") or args.get("commSize")
                if group_size:
                    tp_groups.add(int(group_size))
        
        if tp_groups:
            tp_size = max(tp_groups)  # 取最大值作为 TP 大小
            evidence.append(f"Detected TP size from comm groups: {tp_size}")
            return tp_size
        
        return 1
    
    def _detect_pp_size(self, comm_events: List[Dict], evidence: List[str]) -> int:
        """检测 Pipeline Parallel 大小"""
        # PP 特征：Send/Recv 点对点通信
        pp_ranks = set()
        
        for event in comm_events:
            name = event.get("name", "").lower()
            args = event.get("args", {})
            
            if "send" in name or "recv" in name:
                # 收集参与 P2P 通信的 rank
                src_rank = args.get("src_rank") or args.get("peer")
                dst_rank = args.get("dst_rank") or args.get("peer")
                if src_rank is not None:
                    pp_ranks.add(src_rank)
                if dst_rank is not None:
                    pp_ranks.add(dst_rank)
        
        if len(pp_ranks) > 1:
            # PP size 通常等于参与 P2P 通信的 rank 数
            pp_size = len(pp_ranks)
            evidence.append(f"Detected PP size from P2P communication: {pp_size}")
            return pp_size
        
        return 1
    
    def _detect_dp_size(self, comm_events: List[Dict], evidence: List[str]) -> int:
        """检测 Data Parallel 大小"""
        # DP 特征：跨所有 rank 的 AllReduce（梯度聚合）
        dp_groups = set()
        
        for event in comm_events:
            name = event.get("name", "").lower()
            args = event.get("args", {})
            group_name = str(args.get("group_name", args.get("groupName", ""))).lower()
            
            # DP 特征：
            # 1. 通信组名称包含 "dp" 或 "data"
            # 2. 或者是默认组（无名称）的 AllReduce
            if ("allreduce" in name) and (
                "dp" in group_name or "data" in group_name or not group_name
            ):
                group_size = args.get("group_size") or args.get("commSize")
                if group_size:
                    dp_groups.add(int(group_size))
        
        if dp_groups:
            dp_size = max(dp_groups)
            evidence.append(f"Detected DP size from comm groups: {dp_size}")
            return dp_size
        
        return 1
    
    def _detect_cp_size(self, comm_events: List[Dict], evidence: List[str]) -> int:
        """检测 Context Parallel 大小"""
        for event in comm_events:
            args = event.get("args", {})
            group_name = str(args.get("group_name", args.get("groupName", ""))).lower()
            
            if "cp" in group_name or "context" in group_name or "sequence" in group_name:
                group_size = args.get("group_size") or args.get("commSize")
                if group_size:
                    evidence.append(f"Detected CP size: {group_size}")
                    return int(group_size)
        
        return 1
    
    def _detect_ep_size(self, comm_events: List[Dict], evidence: List[str]) -> int:
        """检测 Expert Parallel 大小（MoE）"""
        for event in comm_events:
            name = event.get("name", "").lower()
            args = event.get("args", {})
            
            # EP 特征：All2All 操作
            if "all2all" in name or "alltoall" in name:
                group_size = args.get("group_size") or args.get("commSize")
                if group_size:
                    evidence.append(f"Detected EP size from All2All: {group_size}")
                    return int(group_size)
        
        return 1
    
    def _detect_zero(
        self,
        comm_events: List[Dict],
        operator_names: List[str],
        evidence: List[str]
    ) -> tuple[bool, Optional[int]]:
        """检测 ZeRO 优化级别"""
        has_reducescatter = False
        has_allgather = False
        
        for event in comm_events:
            name = event.get("name", "").lower()
            if "reducescatter" in name:
                has_reducescatter = True
            if "allgather" in name:
                has_allgather = True
        
        # ZeRO Stage 2/3 特征：ReduceScatter + AllGather 配对
        if has_reducescatter and has_allgather:
            # 简化判断：如果有 AllGather，可能是 Stage 3（参数分片）
            # 如果只有 ReduceScatter，可能是 Stage 2（梯度分片）
            for op_name in operator_names:
                if "zero" in op_name.lower():
                    evidence.append("Detected ZeRO optimizer from operator names")
                    if has_allgather:
                        evidence.append("Detected ZeRO Stage 3 (parameter + gradient sharding)")
                        return True, 3
                    else:
                        evidence.append("Detected ZeRO Stage 2 (gradient sharding)")
                        return True, 2
            
            # 即使没有明确的 ZeRO 算子名称，通信模式也暗示 ZeRO
            if has_reducescatter and has_allgather:
                evidence.append("Detected ZeRO-like sharding pattern (ReduceScatter + AllGather)")
                return True, None
        
        return False, None
    
    def _detect_fsdp(
        self,
        comm_events: List[Dict],
        operator_names: List[str],
        evidence: List[str]
    ) -> bool:
        """检测 FSDP"""
        for op_name in operator_names:
            if "fsdp" in op_name.lower() or "fully_sharded" in op_name.lower():
                evidence.append("Detected FSDP from operator names")
                return True
        
        # FSDP 通信模式类似 ZeRO Stage 3
        # 如果有明确的 FSDP 通信组
        for event in comm_events:
            args = event.get("args", {})
            group_name = str(args.get("group_name", args.get("groupName", ""))).lower()
            if "fsdp" in group_name:
                evidence.append("Detected FSDP from comm group names")
                return True
        
        return False
    
    def _infer_parallel_sizes(
        self,
        world_size: int,
        comm_events: List[Dict],
        evidence: List[str]
    ) -> tuple[int, int, int]:
        """
        推断并行配置（当无法直接检测时）
        
        启发式规则：
        1. 如果有 P2P 通信 → 可能有 PP
        2. 如果 world_size 是 2 的幂 → 可能是纯 DP
        3. 否则尝试分解 world_size
        """
        has_p2p = any(
            "send" in event.get("name", "").lower() or 
            "recv" in event.get("name", "").lower()
            for event in comm_events
        )
        
        if has_p2p:
            # 有 P2P → 可能是 PP
            # 尝试 PP=2, 4, 8...
            for pp_candidate in [2, 4, 8]:
                if world_size % pp_candidate == 0:
                    dp_candidate = world_size // pp_candidate
                    evidence.append(
                        f"Inferred PP={pp_candidate}, DP={dp_candidate} "
                        f"(P2P communication detected)"
                    )
                    return 1, pp_candidate, dp_candidate
        
        # 默认：纯 DP
        evidence.append(f"Inferred pure DP with size={world_size} (default assumption)")
        return 1, 1, world_size
    
    def _calculate_confidence(
        self,
        evidence: List[str],
        comm_events: List[Dict]
    ) -> float:
        """计算检测置信度"""
        if not evidence:
            return 0.0
        
        # 基于证据数量和通信事件数量
        confidence = min(len(evidence) / 5.0, 1.0)  # 最多 5 条证据 = 100% 置信度
        
        if comm_events:
            # 如果有通信事件，提升置信度
            confidence = min(confidence + 0.2, 1.0)
        
        return round(confidence, 2)


def detect_parallel_config_from_loader(loader) -> ParallelConfig:
    """
    从 ProfilingLoader 检测并行配置
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        ParallelConfig
    """
    info = loader.detect()
    world_size = info.rank_count
    
    detector = ParallelDetector(world_size)
    
    # 收集通信事件
    comm_events = []
    try:
        # 从 trace_view.json 或 DB 获取通信事件
        overlap_events = loader.get_overlap_events()
        comm_events = overlap_events.get("hccl", [])
    except Exception as e:
        logger.debug(f"Failed to collect comm events: {e}")
    
    # 收集算子名称
    operator_names = []
    try:
        # 从 DB 获取算子名称
        pass
    except Exception as e:
        logger.debug(f"Failed to collect operator names: {e}")
    
    return detector.detect(comm_events, operator_names)
