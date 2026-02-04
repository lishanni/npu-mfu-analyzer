#!/usr/bin/env python3
"""
示例：分析 Profiling 数据

使用方法：
    python examples/analyze_profiling.py /path/to/profiling_dir
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import ProfilingLoader
from src.analyzers import OverlapCalculator, SlowRankDetector, MFUCalculator
from src.hardware import detect_hardware
from src.pattern_matcher import UniversalPatternMatcher


def main(profiling_path: str):
    """分析 profiling 数据"""
    profiling_dir = Path(profiling_path)
    
    if not profiling_dir.exists():
        print(f"错误：路径不存在 {profiling_dir}")
        return
    
    print(f"分析 Profiling 数据: {profiling_dir}")
    print("=" * 60)
    
    # 1. 加载 profiling 数据
    print("\n[1/5] 加载 Profiling 数据...")
    loader = ProfilingLoader(profiling_dir)
    
    # 2. 硬件检测
    print("\n[2/5] 检测硬件信息...")
    hw_spec = detect_hardware(str(profiling_dir))
    if hw_spec:
        print(f"  检测到: {hw_spec.name} ({hw_spec.variant})")
        print(f"  算力: FP16={hw_spec.peak_tflops_fp16} TFLOPS")
    else:
        print("  未检测到硬件信息，使用默认配置")
    
    # 3. 模式识别
    print("\n[3/5] 识别训练框架和并行策略...")
    matcher = UniversalPatternMatcher()
    # pattern = matcher.detect_from_loader(loader)
    # print(f"  框架: {pattern.framework.framework.value if pattern.framework else 'Unknown'}")
    # print(f"  并行策略: TP={pattern.parallel.tp_size}, PP={pattern.parallel.pp_size}, DP={pattern.parallel.dp_size}")
    
    # 4. 性能分析
    print("\n[4/5] 性能分析...")
    step_trace = loader.get_step_trace()
    if step_trace is not None and not step_trace.empty:
        avg_step_time = step_trace['iteration_time'].mean()
        print(f"  平均 Step 时间: {avg_step_time:.2f} ms")
        
        # 计算 Overlap
        overlap_calc = OverlapCalculator()
        # metrics = overlap_calc.calculate_from_loader(loader)
        # print(f"  通信掩盖率: {metrics.overlap_ratio:.1%}")
    
    # 5. 慢卡检测
    print("\n[5/5] 慢卡检测...")
    detector = SlowRankDetector()
    # result = detector.detect_from_step_trace(step_trace)
    # if result.slow_ranks:
    #     print(f"  检测到慢卡: {result.slow_ranks}")
    # else:
    #     print("  未检测到明显慢卡")
    
    print("\n" + "=" * 60)
    print("分析完成!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python examples/analyze_profiling.py /path/to/profiling_dir")
        sys.exit(1)
    
    main(sys.argv[1])
