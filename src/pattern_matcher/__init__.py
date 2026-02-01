"""
Pattern Matcher 模块

跨框架模式识别，支持自动检测训练框架、并行策略和模型结构
"""

from src.pattern_matcher.framework_detector import (
    FrameworkDetector,
    FrameworkType,
    FrameworkSignature,
    DetectionResult as FrameworkDetectionResult,
    detect_framework_from_loader,
)

from src.pattern_matcher.parallel_detector import (
    ParallelDetector,
    ParallelStrategy,
    ParallelConfig,
    detect_parallel_config_from_loader,
)

from src.pattern_matcher.model_detector import (
    ModelDetector,
    ModelArchitecture,
    ModelConfig,
    detect_model_config_from_loader,
)

from src.pattern_matcher.universal_matcher import (
    UniversalPatternMatcher,
    UniversalPattern,
    detect_pattern_from_loader,
)

__all__ = [
    # Framework Detection
    "FrameworkDetector",
    "FrameworkType",
    "FrameworkSignature",
    "FrameworkDetectionResult",
    "detect_framework_from_loader",
    
    # Parallel Detection
    "ParallelDetector",
    "ParallelStrategy",
    "ParallelConfig",
    "detect_parallel_config_from_loader",
    
    # Model Detection
    "ModelDetector",
    "ModelArchitecture",
    "ModelConfig",
    "detect_model_config_from_loader",
    
    # Universal Matcher
    "UniversalPatternMatcher",
    "UniversalPattern",
    "detect_pattern_from_loader",
]
