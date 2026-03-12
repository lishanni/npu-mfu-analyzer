"""
Hardware 模块 - 硬件规格管理

提供昇腾 NPU 芯片的规格数据库和自动识别功能。
"""

from npu_mfu_analyzer.hardware.registry import (
    NPUSpec,
    DataType,
    HardwareRegistry,
    get_registry,
    detect_hardware,
)

__all__ = [
    "NPUSpec",
    "DataType",
    "HardwareRegistry",
    "get_registry",
    "detect_hardware",
]
