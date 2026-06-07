"""
MFU (Model FLOPS Utilization) 计算器

计算模型在 NPU 上的 FLOPS 利用率。
复用 msprof-analyze 的 MFU 计算逻辑。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import os
import re
import json
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型"""
    FLOAT16 = "FLOAT16"
    BFLOAT16 = "BFLOAT16"
    FLOAT32 = "FLOAT32"
    INT8 = "INT8"


class OperatorType(Enum):
    """算子类型"""
    MATMUL = "MatMul"
    FLASH_ATTENTION = "FlashAttention"
    CONV2D = "Conv2D"
    UNKNOWN = "Unknown"


# 数据类型映射
DTYPE_MAP = {
    "FLOAT": DataType.FLOAT16,
    "FLOAT16": DataType.FLOAT16,
    "BF16": DataType.BFLOAT16,
    "DT_BF16": DataType.BFLOAT16,
    "BFLOAT16": DataType.BFLOAT16,
    "FLOAT32": DataType.FLOAT32,
    "INT8": DataType.INT8,
}

# 算子类型映射
OP_TYPE_MAP = {
    OperatorType.MATMUL: ["MatMulV2", "MatMulV3", "MatMul", "BatchMatMul", "BatchMatMulV2"],
    OperatorType.FLASH_ATTENTION: ["FlashAttentionScore", "FlashAttention", "FusedInferAttentionScore", "FusedAttention"],
    OperatorType.CONV2D: ["Conv2D", "Conv2DBackpropFilter", "Conv2DBackpropInput"],
}


@dataclass
class ChipInfo:
    """芯片信息"""
    aicore_count: int = 0
    aic_frequency: float = 0.0  # MHz
    chip_name: str = ""
    
    # 每周期操作数（根据 Ascend 910B 规格）
    OPS_PER_CYCLE = {
        DataType.FLOAT16: 16 * 16 * 16 * 2,    # Cube Unit: 16x16x16 矩阵乘，乘加算 2 次
        DataType.BFLOAT16: 16 * 16 * 16 * 2,
        DataType.INT8: 16 * 32 * 16 * 2,       # INT8 精度翻倍
    }
    MHZ_TO_HZ = 1_000_000
    
    def get_peak_flops(self, dtype: DataType = DataType.FLOAT16) -> float:
        """
        获取芯片理论峰值 FLOPS
        
        公式: AICore数 × 频率(Hz) × 每周期操作数
        """
        if not self.is_valid():
            return 0.0
        
        ops_per_cycle = self.OPS_PER_CYCLE.get(dtype, self.OPS_PER_CYCLE[DataType.FLOAT16])
        return self.aicore_count * self.aic_frequency * self.MHZ_TO_HZ * ops_per_cycle
    
    def is_valid(self) -> bool:
        return self.aicore_count > 0 and self.aic_frequency > 0
    
    @classmethod
    def from_profiling_path(cls, profiling_path: str) -> "ChipInfo":
        """从 Profiling 数据目录加载芯片信息"""
        chip_info = cls()
        
        # 查找 device_*/info.json.*
        profiling_dir = Path(profiling_path)
        
        for device_dir in profiling_dir.rglob("device_*"):
            if not device_dir.is_dir():
                continue
            
            for info_file in device_dir.glob("info.json*"):
                try:
                    with open(info_file, "r") as f:
                        data = json.load(f)
                    
                    device_info_list = data.get("DeviceInfo", [])
                    if device_info_list:
                        device_info = device_info_list[0]
                        chip_info.aicore_count = int(device_info.get("ai_core_num", 0))
                        
                        freq_str = device_info.get("aic_frequency", "0")
                        if isinstance(freq_str, str):
                            freq_str = re.sub(r"[^\d.]", "", freq_str)
                        chip_info.aic_frequency = float(freq_str) if freq_str else 0.0
                        
                        chip_info.chip_name = device_info.get("soc_name", "")
                        
                        if chip_info.is_valid():
                            logger.info(
                                f"Loaded chip info: {chip_info.chip_name}, "
                                f"AICore={chip_info.aicore_count}, freq={chip_info.aic_frequency}MHz"
                            )
                            return chip_info
                except Exception as e:
                    logger.debug(f"Failed to load chip info from {info_file}: {e}")
        
        logger.warning("Could not load chip info from profiling data")
        return chip_info
    
    @classmethod
    def default_ascend_910b(cls) -> "ChipInfo":
        """默认 Ascend 910B 配置"""
        return cls(
            aicore_count=32,       # 910B 有 32 个 AICore
            aic_frequency=1800.0,  # 1.8 GHz
            chip_name="Ascend 910B",
        )


@dataclass
class OperatorMFU:
    """算子 MFU 信息"""
    name: str
    op_type: OperatorType
    flops: float = 0.0           # 实际 FLOPs
    duration_ns: float = 0.0     # 执行时间（纳秒）
    mfu: float = 0.0             # MFU 值 (0~1)
    dtype: DataType = DataType.FLOAT16
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type.value,
            "flops": self.flops,
            "duration_ms": self.duration_ns / 1e6,
            "mfu": self.mfu,
            "dtype": self.dtype.value,
        }


@dataclass
class MFUMetrics:
    """MFU 分析指标"""
    # 整体指标
    overall_mfu: float = 0.0           # 整体 MFU
    peak_flops: float = 0.0            # 芯片峰值 FLOPS
    actual_flops: float = 0.0          # 实际 FLOPS
    total_duration_ns: float = 0.0     # 总执行时间
    
    # 各类算子 MFU
    matmul_mfu: float = 0.0
    attention_mfu: float = 0.0
    
    # Top 算子
    top_operators: List[OperatorMFU] = field(default_factory=list)
    low_mfu_operators: List[OperatorMFU] = field(default_factory=list)
    
    # 统计信息
    operator_count: int = 0
    matmul_count: int = 0
    attention_count: int = 0
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## MFU 分析",
            "",
            "### 整体指标",
            f"- 整体 MFU: {self.overall_mfu * 100:.1f}%",
            f"- 芯片峰值: {self.peak_flops / 1e12:.1f} TFLOPS",
            f"- 实际算力: {self.actual_flops / 1e12:.3f} TFLOPS",
            f"- 总执行时间: {self.total_duration_ns / 1e6:.2f} ms",
            "",
            "### 各类算子 MFU",
            f"- MatMul MFU: {self.matmul_mfu * 100:.1f}% ({self.matmul_count} 个)",
            f"- Attention MFU: {self.attention_mfu * 100:.1f}% ({self.attention_count} 个)",
        ]
        
        if self.top_operators:
            lines.append("")
            lines.append("### Top 5 高 MFU 算子")
            for i, op in enumerate(self.top_operators[:5], 1):
                lines.append(f"{i}. {op.name}: MFU={op.mfu*100:.1f}%, {op.duration_ns/1e6:.2f}ms")
        
        if self.low_mfu_operators:
            lines.append("")
            lines.append("### 低 MFU 算子（需优化）")
            for op in self.low_mfu_operators[:5]:
                lines.append(f"- {op.name}: MFU={op.mfu*100:.1f}%")
        
        return "\n".join(lines)


class MFUCalculator:
    """
    MFU 计算器
    
    MFU (Model FLOPS Utilization) = 实际 FLOPS / 芯片峰值 FLOPS
    
    计算公式：
    - MatMul FLOPS = M × N × K × 2（乘加算 2 次操作）
    - FlashAttention FLOPS = 2 × B × N × S × S × (D + D')
    - MFU = FLOPS / (duration_s) / peak_flops
    """
    
    def __init__(self, chip_info: Optional[ChipInfo] = None):
        """
        Args:
            chip_info: 芯片信息，如果不提供则尝试从 Profiling 数据加载
        """
        self.chip_info = chip_info
    
    def calculate_matmul_flops(
        self, 
        m: int, 
        n: int, 
        k: int,
    ) -> float:
        """
        计算 MatMul FLOPs
        
        公式: M × N × K × 2（乘加算 2 次操作）
        """
        return m * n * k * 2
    
    def calculate_attention_flops(
        self,
        batch: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        head_dim: int,
        is_causal: bool = False,
    ) -> float:
        """
        计算 FlashAttention FLOPs
        
        公式: 2 × B × N × Sq × Skv × 2D
        - 如果 is_causal=True（下三角 mask），实际 FLOPs 减半
        """
        full_flops = 2 * batch * num_heads * seq_len_q * seq_len_kv * 2 * head_dim
        if is_causal and seq_len_q == seq_len_kv:
            return full_flops * 0.5
        return full_flops
    
    def calculate_operator_mfu(
        self,
        flops: float,
        duration_ns: float,
        dtype: DataType = DataType.FLOAT16,
    ) -> float:
        """
        计算单个算子的 MFU
        
        Args:
            flops: 算子的 FLOPs
            duration_ns: 执行时间（纳秒）
            dtype: 数据类型
            
        Returns:
            MFU 值 (0~1)
        """
        if not self.chip_info or not self.chip_info.is_valid():
            return 0.0
        
        if duration_ns <= 0 or flops <= 0:
            return 0.0
        
        peak_flops = self.chip_info.get_peak_flops(dtype)
        duration_s = duration_ns / 1e9
        actual_flops_per_second = flops / duration_s
        
        return actual_flops_per_second / peak_flops
    
    def analyze_operators(
        self,
        operators_df: pd.DataFrame,
        name_col: str = "name",
        dur_col: str = "dur",
        shapes_col: str = "input_shapes",
        dtypes_col: str = "input_types",
    ) -> MFUMetrics:
        """
        分析算子 MFU
        
        Args:
            operators_df: 算子数据 DataFrame
            name_col: 算子名称列
            dur_col: 执行时间列（纳秒）
            shapes_col: 输入形状列
            dtypes_col: 输入数据类型列
            
        Returns:
            MFUMetrics
        """
        metrics = MFUMetrics()
        
        if operators_df.empty or not self.chip_info or not self.chip_info.is_valid():
            return metrics
        
        metrics.peak_flops = self.chip_info.get_peak_flops()
        
        operator_mfus = []
        
        for _, row in operators_df.iterrows():
            name = str(row.get(name_col, ""))
            duration_ns = float(row.get(dur_col, 0))
            
            # 识别算子类型
            op_type = self._identify_operator_type(name)
            if op_type == OperatorType.UNKNOWN:
                continue
            
            # 解析形状计算 FLOPs
            shapes_str = str(row.get(shapes_col, ""))
            dtypes_str = str(row.get(dtypes_col, ""))
            
            try:
                flops = self._calculate_flops(op_type, shapes_str, row)
                if flops <= 0:
                    continue
                
                dtype = self._parse_dtype(dtypes_str)
                mfu = self.calculate_operator_mfu(flops, duration_ns, dtype)
                
                op_mfu = OperatorMFU(
                    name=name,
                    op_type=op_type,
                    flops=flops,
                    duration_ns=duration_ns,
                    mfu=mfu,
                    dtype=dtype,
                )
                operator_mfus.append(op_mfu)
                
                metrics.actual_flops += flops
                metrics.total_duration_ns += duration_ns
                metrics.operator_count += 1
                
                if op_type == OperatorType.MATMUL:
                    metrics.matmul_count += 1
                elif op_type == OperatorType.FLASH_ATTENTION:
                    metrics.attention_count += 1
                    
            except Exception as e:
                logger.debug(f"Failed to calculate MFU for {name}: {e}")
        
        # 计算整体 MFU
        if metrics.total_duration_ns > 0 and metrics.peak_flops > 0:
            duration_s = metrics.total_duration_ns / 1e9
            metrics.overall_mfu = (metrics.actual_flops / duration_s) / metrics.peak_flops
        
        # 计算各类算子 MFU
        matmul_ops = [op for op in operator_mfus if op.op_type == OperatorType.MATMUL]
        attention_ops = [op for op in operator_mfus if op.op_type == OperatorType.FLASH_ATTENTION]
        
        if matmul_ops:
            metrics.matmul_mfu = np.mean([op.mfu for op in matmul_ops])
        if attention_ops:
            metrics.attention_mfu = np.mean([op.mfu for op in attention_ops])
        
        # Top 算子排序
        operator_mfus.sort(key=lambda x: x.mfu, reverse=True)
        metrics.top_operators = operator_mfus[:10]
        
        # 低 MFU 算子
        low_mfu_threshold = 0.3
        metrics.low_mfu_operators = [
            op for op in operator_mfus 
            if op.mfu < low_mfu_threshold and op.mfu > 0
        ][:10]
        
        return metrics
    
    def _identify_operator_type(self, name: str) -> OperatorType:
        """识别算子类型"""
        name_lower = name.lower()
        
        for op_type, patterns in OP_TYPE_MAP.items():
            for pattern in patterns:
                if pattern.lower() in name_lower:
                    return op_type
        
        return OperatorType.UNKNOWN
    
    def _calculate_flops(
        self, 
        op_type: OperatorType, 
        shapes_str: str,
        row: pd.Series,
    ) -> float:
        """计算算子 FLOPs"""
        if op_type == OperatorType.MATMUL:
            return self._calculate_matmul_flops_from_shapes(shapes_str, row)
        elif op_type == OperatorType.FLASH_ATTENTION:
            return self._calculate_attention_flops_from_shapes(shapes_str, row)
        return 0.0
    
    def _calculate_matmul_flops_from_shapes(
        self, 
        shapes_str: str,
        row: pd.Series,
    ) -> float:
        """从形状字符串计算 MatMul FLOPs"""
        try:
            # 尝试解析 output_shapes 获取 M, N
            output_shapes_str = str(row.get("output_shapes", ""))
            
            shapes = self._parse_shapes(shapes_str)
            output_shapes = self._parse_shapes(output_shapes_str)
            
            if len(shapes) >= 2 and len(output_shapes) >= 1:
                # 从输出形状获取 M, N
                if len(output_shapes[0]) == 2:
                    m, n = output_shapes[0]
                    # 从输入推断 K
                    if len(shapes[0]) == 2:
                        k = shapes[0][1] if shapes[0][0] == m else shapes[0][0]
                        return self.calculate_matmul_flops(m, n, k)
            
            # 简化估算：假设是方阵
            if len(shapes) >= 1 and len(shapes[0]) >= 2:
                m = shapes[0][0]
                k = shapes[0][1]
                n = k  # 假设方阵
                return self.calculate_matmul_flops(m, n, k)
                
        except Exception as e:
            logger.debug(f"Failed to parse MatMul shapes: {e}")
        
        return 0.0
    
    def _calculate_attention_flops_from_shapes(
        self, 
        shapes_str: str,
        row: pd.Series,
    ) -> float:
        """从形状字符串计算 Attention FLOPs"""
        try:
            shapes = self._parse_shapes(shapes_str)
            
            if len(shapes) >= 3:
                # Q, K, V shapes
                q_shape = shapes[0]
                k_shape = shapes[1]
                
                # 假设 BNSD layout
                if len(q_shape) == 4:
                    b, n, s, d = q_shape
                    kv_s = k_shape[2] if len(k_shape) == 4 else s
                    return self.calculate_attention_flops(b, n, s, kv_s, d, is_causal=True)
                    
        except Exception as e:
            logger.debug(f"Failed to parse Attention shapes: {e}")
        
        return 0.0
    
    def _parse_shapes(self, shapes_str: str) -> List[List[int]]:
        """解析形状字符串"""
        # 清理引号、方括号等非数字字符
        cleaned = shapes_str.strip().strip('[]').replace('"', '').replace("'", "")
        shapes = []
        for shape_part in cleaned.split(";"):
            shape_part = shape_part.strip()
            if shape_part:
                try:
                    shape = [int(dim.strip()) for dim in shape_part.split(",") if dim.strip()]
                    shapes.append(shape)
                except ValueError:
                    shapes.append([])
            else:
                shapes.append([])
        return shapes
    
    def _parse_dtype(self, dtypes_str: str) -> DataType:
        """解析数据类型"""
        if not dtypes_str:
            return DataType.FLOAT16
        
        first_type = dtypes_str.split(";")[0].strip().upper()
        return DTYPE_MAP.get(first_type, DataType.FLOAT16)
