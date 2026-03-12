"""
Hardware Registry - 硬件规格注册表

提供昇腾 NPU 芯片的完整规格数据库，支持：
1. 自动识别芯片型号（从 Profiling 数据）
2. 手动指定芯片型号
3. 计算理论峰值性能
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型"""
    FP16 = "FP16"
    BF16 = "BF16"
    FP32 = "FP32"
    INT8 = "INT8"
    INT4 = "INT4"


@dataclass
class NPUSpec:
    """
    NPU 芯片完整规格
    
    包含算力、带宽、缓存等所有性能相关参数
    """
    # 基本信息
    name: str = ""                      # "Atlas 800T A2 910B"
    variant: str = ""                   # "280T" / "313T" / "376T"
    
    # 计算单元
    aicore_count: int = 0               # AICore 数量
    aicore_freq_mhz: int = 0            # AICore 频率 (MHz)
    cube_ops_per_cycle: int = 4096      # Cube Unit 每周期操作数 (FP16)
    vector_ops_per_cycle: int = 256     # Vector Unit 每周期操作数
    
    # 内存带宽
    hbm_bandwidth_gbps: float = 0.0     # HBM 带宽 (GB/s)
    hbm_capacity_gb: int = 0            # HBM 容量 (GB)
    
    # 互联带宽
    hccs_bandwidth_gbps: float = 0.0    # HCCS 单链路带宽 (GB/s)
    hccs_links: int = 0                 # HCCS 链路数量
    pcie_bandwidth_gbps: float = 0.0    # PCIe 带宽 (GB/s)
    
    # 缓存
    l2_cache_mb: int = 0                # L2 Cache 大小 (MB)
    
    # 预计算峰值（可选，用于验证）
    peak_tflops_fp16: float = 0.0       # 标称 FP16 峰值 TFLOPS
    
    # 匹配模式
    soc_name_patterns: List[str] = field(default_factory=list)
    
    def get_peak_tflops(self, dtype: DataType = DataType.FP16) -> float:
        """
        计算理论峰值 TFLOPS
        
        公式: AICore数 × 频率(Hz) × 每周期操作数 / 1e12
        """
        if self.aicore_count == 0 or self.aicore_freq_mhz == 0:
            return 0.0
        
        # 不同精度的每周期操作数倍率
        dtype_multiplier = {
            DataType.FP16: 1.0,
            DataType.BF16: 1.0,
            DataType.FP32: 0.5,     # FP32 算力减半
            DataType.INT8: 2.0,     # INT8 算力翻倍
            DataType.INT4: 4.0,     # INT4 算力 4 倍
        }
        
        multiplier = dtype_multiplier.get(dtype, 1.0)
        ops_per_cycle = self.cube_ops_per_cycle * multiplier
        
        freq_hz = self.aicore_freq_mhz * 1e6
        peak_flops = self.aicore_count * freq_hz * ops_per_cycle
        
        return peak_flops / 1e12  # 转换为 TFLOPS
    
    def get_total_hccs_bandwidth_gbps(self) -> float:
        """获取 HCCS 总带宽"""
        return self.hccs_bandwidth_gbps * self.hccs_links
    
    def get_arithmetic_intensity_threshold(self, dtype: DataType = DataType.FP16) -> float:
        """
        计算 Roofline 模型的算术强度阈值
        
        AI_threshold = Peak_TFLOPS / HBM_BW
        当 AI > threshold 时为 Compute-Bound，否则为 Memory-Bound
        """
        if self.hbm_bandwidth_gbps == 0:
            return float('inf')
        
        peak_tflops = self.get_peak_tflops(dtype)
        # 转换单位: TFLOPS / (GB/s) = FLOPs/Byte
        return (peak_tflops * 1e12) / (self.hbm_bandwidth_gbps * 1e9)
    
    def is_valid(self) -> bool:
        """检查规格是否有效"""
        return self.aicore_count > 0 and self.aicore_freq_mhz > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "variant": self.variant,
            "aicore_count": self.aicore_count,
            "aicore_freq_mhz": self.aicore_freq_mhz,
            "cube_ops_per_cycle": self.cube_ops_per_cycle,
            "vector_ops_per_cycle": self.vector_ops_per_cycle,
            "hbm_bandwidth_gbps": self.hbm_bandwidth_gbps,
            "hbm_capacity_gb": self.hbm_capacity_gb,
            "hccs_bandwidth_gbps": self.hccs_bandwidth_gbps,
            "hccs_links": self.hccs_links,
            "pcie_bandwidth_gbps": self.pcie_bandwidth_gbps,
            "l2_cache_mb": self.l2_cache_mb,
            "peak_tflops_fp16": self.peak_tflops_fp16 or self.get_peak_tflops(),
        }
    
    def to_summary(self) -> str:
        """生成摘要文本"""
        lines = [
            f"芯片型号: {self.name} {self.variant}",
            f"AICore: {self.aicore_count} 个 @ {self.aicore_freq_mhz} MHz",
            f"峰值算力: {self.get_peak_tflops():.1f} TFLOPS (FP16)",
            f"HBM: {self.hbm_capacity_gb} GB @ {self.hbm_bandwidth_gbps:.0f} GB/s",
            f"HCCS: {self.hccs_links} links @ {self.hccs_bandwidth_gbps:.0f} GB/s each",
            f"L2 Cache: {self.l2_cache_mb} MB",
        ]
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPUSpec":
        """从字典创建"""
        return cls(
            name=data.get("name", ""),
            variant=data.get("variant", ""),
            aicore_count=data.get("aicore_count", 0),
            aicore_freq_mhz=data.get("aicore_freq_mhz", 0),
            cube_ops_per_cycle=data.get("cube_ops_per_cycle", 4096),
            vector_ops_per_cycle=data.get("vector_ops_per_cycle", 256),
            hbm_bandwidth_gbps=data.get("hbm_bandwidth_gbps", 0.0),
            hbm_capacity_gb=data.get("hbm_capacity_gb", 0),
            hccs_bandwidth_gbps=data.get("hccs_bandwidth_gbps", 0.0),
            hccs_links=data.get("hccs_links", 0),
            pcie_bandwidth_gbps=data.get("pcie_bandwidth_gbps", 0.0),
            l2_cache_mb=data.get("l2_cache_mb", 0),
            peak_tflops_fp16=data.get("peak_tflops_fp16", 0.0),
            soc_name_patterns=data.get("soc_name_patterns", []),
        )


class HardwareRegistry:
    """
    硬件规格注册表
    
    功能：
    1. 加载预定义的芯片规格（从 YAML 文件）
    2. 自动识别芯片型号（从 Profiling 数据）
    3. 支持手动覆盖
    4. 根据 AICore 数量推断型号
    """
    
    # 默认规格目录
    DEFAULT_SPECS_DIR = Path(__file__).parent / "specs"
    
    def __init__(self, specs_dir: Optional[Path] = None):
        """
        Args:
            specs_dir: 规格文件目录，默认使用内置目录
        """
        self.specs_dir = specs_dir or self.DEFAULT_SPECS_DIR
        self._specs: Dict[str, NPUSpec] = {}
        self._loaded = False
    
    def _ensure_loaded(self):
        """确保规格已加载"""
        if not self._loaded:
            self._load_specs()
            self._loaded = True
    
    def _load_specs(self):
        """从 YAML 文件加载规格"""
        if not self.specs_dir.exists():
            logger.warning(f"Specs directory not found: {self.specs_dir}")
            return
        
        for yaml_file in self.specs_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                
                chips = data.get("chips", [])
                for chip_data in chips:
                    spec = NPUSpec.from_dict(chip_data)
                    key = self._make_key(spec.name, spec.variant)
                    self._specs[key] = spec
                    logger.debug(f"Loaded spec: {key}")
                    
            except Exception as e:
                logger.error(f"Failed to load specs from {yaml_file}: {e}")
        
        logger.info(f"Loaded {len(self._specs)} chip specifications")
    
    def _make_key(self, name: str, variant: str) -> str:
        """生成规格键"""
        return f"{name}:{variant}".lower().replace(" ", "_")
    
    def get_spec(self, name: str, variant: str = "") -> Optional[NPUSpec]:
        """
        获取指定芯片规格
        
        Args:
            name: 芯片名称
            variant: 变体（如 280T, 313T）
        """
        self._ensure_loaded()
        key = self._make_key(name, variant)
        return self._specs.get(key)
    
    def get_spec_by_aicore_count(self, aicore_count: int, freq_mhz: int = 0) -> Optional[NPUSpec]:
        """
        根据 AICore 数量推断芯片型号
        
        Args:
            aicore_count: AICore 数量
            freq_mhz: 频率（可选，用于更精确匹配）
        """
        self._ensure_loaded()
        
        # AICore 数量到型号的映射
        aicore_mapping = {
            20: ("Atlas 800T A2 910B", "280T"),
            24: ("Atlas 800T A2 910B", "313T"),
            32: ("Atlas 800T A2 910B", "376T"),
            40: ("Atlas 900 A2 910B Pro", "default"),
        }
        
        if aicore_count in aicore_mapping:
            name, variant = aicore_mapping[aicore_count]
            spec = self.get_spec(name, variant)
            if spec:
                # 如果提供了频率，更新规格
                if freq_mhz > 0 and freq_mhz != spec.aicore_freq_mhz:
                    spec = NPUSpec.from_dict(spec.to_dict())
                    spec.aicore_freq_mhz = freq_mhz
                return spec
        
        # 未找到精确匹配，创建通用规格
        logger.warning(f"Unknown AICore count: {aicore_count}, using generic spec")
        return NPUSpec(
            name="Unknown NPU",
            variant=f"{aicore_count}AICore",
            aicore_count=aicore_count,
            aicore_freq_mhz=freq_mhz or 1800,
            hbm_bandwidth_gbps=1600,
            hbm_capacity_gb=64,
        )
    
    def get_spec_by_soc_name(self, soc_name: str) -> Optional[NPUSpec]:
        """
        根据 SOC 名称匹配芯片规格
        
        Args:
            soc_name: 从 Profiling 数据读取的 soc_name
        """
        self._ensure_loaded()
        
        if not soc_name:
            return None
        
        soc_name_lower = soc_name.lower()
        
        for spec in self._specs.values():
            for pattern in spec.soc_name_patterns:
                if pattern.lower() in soc_name_lower:
                    return spec
        
        return None
    
    def detect_from_profiling(self, profiling_path: str) -> Optional[NPUSpec]:
        """
        从 Profiling 数据自动检测芯片规格
        
        Args:
            profiling_path: Profiling 数据目录
        """
        self._ensure_loaded()
        
        profiling_dir = Path(profiling_path)
        
        # 查找 device_*/info.json.*
        for info_file in profiling_dir.rglob("device_*/info.json*"):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                device_info_list = data.get("DeviceInfo", [])
                if not device_info_list:
                    continue
                
                device_info = device_info_list[0]
                
                # 提取关键信息
                aicore_count = int(device_info.get("ai_core_num", 0))
                freq_str = device_info.get("aic_frequency", "0")
                if isinstance(freq_str, str):
                    freq_str = re.sub(r"[^\d.]", "", freq_str)
                freq_mhz = int(float(freq_str)) if freq_str else 0
                soc_name = device_info.get("soc_name", "")
                
                if aicore_count == 0:
                    continue
                
                # 优先使用 SOC 名称匹配
                if soc_name:
                    spec = self.get_spec_by_soc_name(soc_name)
                    if spec:
                        logger.info(f"Detected chip by soc_name: {spec.name} {spec.variant}")
                        return spec
                
                # 使用 AICore 数量推断
                spec = self.get_spec_by_aicore_count(aicore_count, freq_mhz)
                if spec:
                    logger.info(f"Detected chip by aicore_count: {spec.name} {spec.variant}")
                    return spec
                    
            except Exception as e:
                logger.debug(f"Failed to read info from {info_file}: {e}")
        
        logger.warning("Could not detect chip spec from profiling data")
        return None
    
    def list_specs(self) -> List[NPUSpec]:
        """列出所有已注册的规格"""
        self._ensure_loaded()
        return list(self._specs.values())
    
    def register_spec(self, spec: NPUSpec):
        """注册自定义规格"""
        key = self._make_key(spec.name, spec.variant)
        self._specs[key] = spec
        logger.info(f"Registered custom spec: {key}")
    
    @classmethod
    def get_default_910b(cls) -> NPUSpec:
        """获取默认 910B 规格（280T）"""
        return NPUSpec(
            name="Atlas 800T A2 910B",
            variant="280T",
            aicore_count=20,
            aicore_freq_mhz=1800,
            cube_ops_per_cycle=4096,
            vector_ops_per_cycle=256,
            hbm_bandwidth_gbps=1600,
            hbm_capacity_gb=64,
            hccs_bandwidth_gbps=56,
            hccs_links=8,
            pcie_bandwidth_gbps=64,
            l2_cache_mb=192,
            peak_tflops_fp16=280,
        )


# 全局单例
_registry: Optional[HardwareRegistry] = None


def get_registry() -> HardwareRegistry:
    """获取全局硬件注册表实例"""
    global _registry
    if _registry is None:
        _registry = HardwareRegistry()
    return _registry


def detect_hardware(profiling_path: str) -> NPUSpec:
    """
    便捷函数：从 Profiling 数据检测硬件规格
    
    Args:
        profiling_path: Profiling 数据目录
        
    Returns:
        NPUSpec，如果检测失败则返回默认 910B 规格
    """
    registry = get_registry()
    spec = registry.detect_from_profiling(profiling_path)
    if spec is None:
        logger.warning("Using default 910B spec")
        spec = HardwareRegistry.get_default_910b()
    return spec
