"""
Roofline 模块

提供性能天花板分析和假设场景模拟
"""

from .roofline_model import (
    RooflineModeler,
    RooflineAnalysis,
    RooflinePoint,
    OperatorProfile,
    HardwareSpec,
    BoundType,
    PrecisionType,
)

from .whatif_simulator import (
    WhatIfSimulator,
    WhatIfScenario,
    SimulationResult,
    CurrentState,
    ScenarioType,
)

__all__ = [
    # Roofline Model
    "RooflineModeler",
    "RooflineAnalysis",
    "RooflinePoint",
    "OperatorProfile",
    "HardwareSpec",
    "BoundType",
    "PrecisionType",
    # What-if Simulator
    "WhatIfSimulator",
    "WhatIfScenario",
    "SimulationResult",
    "CurrentState",
    "ScenarioType",
]
