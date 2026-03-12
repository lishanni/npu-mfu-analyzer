"""
Roofline & What-if 集成测试

验证 Phase 9 的性能天花板分析和假设模拟功能
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from npu_mfu_analyzer.roofline import (
    RooflineModeler,
    RooflineAnalysis,
    OperatorProfile,
    PrecisionType,
    BoundType,
    WhatIfSimulator,
    CurrentState,
    ScenarioType,
)


def test_roofline_basics():
    """测试 Roofline 基础功能"""
    print("=" * 60)
    print("测试 1: Roofline 基础功能")
    print("=" * 60)
    
    modeler = RooflineModeler(hardware_name="atlas_a2_280t")
    
    print(f"\n硬件: {modeler.hardware.name}")
    print(f"FP16 峰值算力: {modeler.hardware.fp16_tflops} TFLOPS")
    print(f"HBM 带宽: {modeler.hardware.hbm_bandwidth_gbps} GB/s")
    
    # 计算脊点
    ridge = modeler.get_ridge_point(PrecisionType.FP16)
    print(f"\n脊点 (Ridge Point): {ridge:.1f} FLOP/Byte")
    print("  - 计算强度 < 脊点 → 内存受限")
    print("  - 计算强度 > 脊点 → 计算受限")
    
    # 测试不同计算强度
    print("\n不同计算强度下的理论天花板:")
    for ai in [1, 10, 50, 100, 500]:
        compute_roof = modeler.get_compute_roof(PrecisionType.FP16)
        memory_roof = modeler.get_memory_roof(ai)
        effective_roof = min(compute_roof, memory_roof)
        bound = "内存受限" if memory_roof < compute_roof else "计算受限"
        print(f"  AI={ai:3} FLOP/Byte → {effective_roof:6.1f} TFLOPS ({bound})")


def test_operator_analysis():
    """测试算子级 Roofline 分析"""
    print("\n" + "=" * 60)
    print("测试 2: 算子级 Roofline 分析")
    print("=" * 60)
    
    modeler = RooflineModeler(hardware_name="atlas_a2_280t")
    
    # 模拟几个典型算子
    operators = [
        OperatorProfile(
            name="MatMul_4096x4096",
            flops=2 * 4096 * 4096 * 4096,  # 2*M*N*K
            memory_bytes=3 * 4096 * 4096 * 2,  # A + B + C, FP16
            duration_us=500,
            precision=PrecisionType.FP16,
        ),
        OperatorProfile(
            name="LayerNorm",
            flops=4096 * 4096 * 5,  # 简化估算
            memory_bytes=4096 * 4096 * 2 * 3,  # 读入写出
            duration_us=50,
            precision=PrecisionType.FP16,
        ),
        OperatorProfile(
            name="Softmax",
            flops=4096 * 4096 * 3,
            memory_bytes=4096 * 4096 * 2 * 2,
            duration_us=30,
            precision=PrecisionType.FP16,
        ),
        OperatorProfile(
            name="GEMM_Large",
            flops=2 * 8192 * 8192 * 8192,
            memory_bytes=3 * 8192 * 8192 * 2,
            duration_us=2000,
            precision=PrecisionType.FP16,
        ),
    ]
    
    # 分析
    analysis = modeler.analyze(operators, PrecisionType.FP16)
    
    print(f"\n分析结果:")
    print(f"  整体受限类型: {analysis.overall_bound.value}")
    print(f"  平均效率: {analysis.avg_efficiency:.1f}%")
    print(f"  脊点: {analysis.ridge_point:.1f} FLOP/Byte")
    
    print(f"\n各算子分析:")
    for point in analysis.points:
        print(f"  {point.name}:")
        print(f"    计算强度: {point.arithmetic_intensity:.1f} FLOP/Byte")
        print(f"    实际性能: {point.achieved_performance:.1f} TFLOPS")
        print(f"    理论天花板: {point.theoretical_peak:.1f} TFLOPS")
        print(f"    效率: {point.efficiency:.1f}%")
        print(f"    受限类型: {point.bound_type.value}")
    
    print(f"\n优化建议:")
    for suggestion in analysis.suggestions:
        print(f"  - {suggestion}")


def test_theoretical_mfu():
    """测试理论 MFU 估算"""
    print("\n" + "=" * 60)
    print("测试 3: 理论 MFU 上限估算")
    print("=" * 60)
    
    modeler = RooflineModeler(hardware_name="atlas_a2_280t")
    
    # 模拟 7B 模型
    # 假设参数：
    # - 7B 参数，每次前向+反向约 6 * 7B = 42B FLOPS
    # - 内存访问：参数 + 激活 + 梯度
    model_flops = 42e12  # 42 TFLOPS per step
    model_memory = 50e9  # 50 GB per step
    step_time_ms = 500
    num_devices = 8
    
    result = modeler.estimate_theoretical_mfu(
        model_flops=model_flops,
        model_memory_bytes=model_memory,
        step_time_ms=step_time_ms,
        num_devices=num_devices,
        precision=PrecisionType.BF16,
    )
    
    print(f"\n7B 模型 @ 8x Atlas A2 (280T):")
    print(f"  模型 FLOPS/step: {model_flops/1e12:.1f} TFLOPS")
    print(f"  内存访问/step: {model_memory/1e9:.1f} GB")
    print(f"  Step 时间: {step_time_ms} ms")
    print(f"\n分析结果:")
    print(f"  计算强度: {result['arithmetic_intensity']:.1f} FLOP/Byte")
    print(f"  脊点: {result['ridge_point']:.1f} FLOP/Byte")
    print(f"  受限类型: {result['bound_description']}")
    print(f"\n性能指标:")
    print(f"  计算天花板: {result['compute_roof_tflops']:.0f} TFLOPS")
    print(f"  内存天花板: {result['memory_roof_tflops']:.0f} TFLOPS")
    print(f"  理论天花板: {result['theoretical_roof_tflops']:.0f} TFLOPS")
    print(f"  实际性能: {result['actual_tflops']:.1f} TFLOPS")
    print(f"\nMFU 分析:")
    print(f"  实际 MFU: {result['actual_mfu_percent']:.1f}%")
    print(f"  理论最大 MFU: {result['theoretical_max_mfu_percent']:.1f}%")
    print(f"  相对效率: {result['roof_efficiency_percent']:.1f}%")
    print(f"  MFU 差距: {result['mfu_gap_percent']:.1f}%")


def test_whatif_parallel():
    """测试 What-if 并行度变化"""
    print("\n" + "=" * 60)
    print("测试 4: What-if 并行度变化模拟")
    print("=" * 60)
    
    state = CurrentState(
        hardware_name="Atlas A2 (280T)",
        num_devices=8,
        tp_size=1,
        pp_size=1,
        dp_size=8,
        model_params_b=7.0,
        step_time_ms=500,
        mfu_percent=40,
        overlap_ratio=0.7,
        comm_time_ratio=0.2,
    )
    
    simulator = WhatIfSimulator(state)
    
    print(f"\n当前配置: TP{state.tp_size}_PP{state.pp_size}_DP{state.dp_size}")
    print(f"当前 Step 时间: {state.step_time_ms}ms")
    print(f"当前 MFU: {state.mfu_percent}%")
    
    # 测试不同并行配置
    configs = [
        (2, 1, 4),   # TP=2
        (4, 1, 2),   # TP=4
        (1, 2, 4),   # PP=2
        (2, 2, 2),   # TP=2, PP=2
    ]
    
    print(f"\n并行配置变化预测:")
    for tp, pp, dp in configs:
        scenario = simulator.simulate_parallel_change(tp, pp, dp)
        print(f"\n  {scenario.name}:")
        print(f"    预测加速: {scenario.predicted_speedup:.2f}x")
        print(f"    预测 MFU: {scenario.predicted_mfu_percent:.1f}%")
        print(f"    置信度: {scenario.confidence:.0%}")
        if scenario.benefits:
            print(f"    优势: {scenario.benefits[0]}")
        if scenario.risks:
            print(f"    风险: {scenario.risks[0]}")


def test_whatif_hardware():
    """测试 What-if 硬件升级"""
    print("\n" + "=" * 60)
    print("测试 5: What-if 硬件升级模拟")
    print("=" * 60)
    
    state = CurrentState(
        hardware_name="Atlas A2 (280T)",
        num_devices=8,
        step_time_ms=500,
        mfu_percent=40,
    )
    
    simulator = WhatIfSimulator(state)
    
    print(f"\n当前硬件: {state.hardware_name}")
    
    for new_hw in ["313T", "376T"]:
        scenario = simulator.simulate_hardware_upgrade(new_hw)
        print(f"\n  升级到 {new_hw}:")
        print(f"    预测加速: {scenario.predicted_speedup:.2f}x")
        print(f"    预测 Step 时间: {scenario.predicted_step_time_ms:.0f}ms")
        if scenario.benefits:
            for b in scenario.benefits:
                print(f"    ✓ {b}")


def test_whatif_optimization():
    """测试 What-if 优化措施"""
    print("\n" + "=" * 60)
    print("测试 6: What-if 优化措施模拟")
    print("=" * 60)
    
    state = CurrentState(
        step_time_ms=500,
        mfu_percent=35,
        overlap_ratio=0.6,
        comm_time_ratio=0.25,
        precision="bf16",
    )
    
    simulator = WhatIfSimulator(state)
    
    optimizations = [
        "gradient_accumulation",
        "overlap_optimization",
        "operator_fusion",
    ]
    
    print(f"\n当前状态:")
    print(f"  Step 时间: {state.step_time_ms}ms")
    print(f"  MFU: {state.mfu_percent}%")
    print(f"  通信掩盖率: {state.overlap_ratio*100:.0f}%")
    
    print(f"\n优化措施预测:")
    for opt in optimizations:
        scenario = simulator.simulate_optimization(opt)
        print(f"\n  {scenario.name}:")
        print(f"    预测加速: {scenario.predicted_speedup:.2f}x")
        print(f"    置信度: {scenario.confidence:.0%}")
        if scenario.benefits:
            print(f"    优势:")
            for b in scenario.benefits[:2]:
                print(f"      - {b}")


def test_full_simulation():
    """测试完整模拟流程"""
    print("\n" + "=" * 60)
    print("测试 7: 完整 What-if 模拟")
    print("=" * 60)
    
    state = CurrentState(
        hardware_name="Atlas A2 (280T)",
        num_devices=8,
        peak_tflops_per_device=280,
        tp_size=1,
        pp_size=1,
        dp_size=8,
        model_params_b=7.0,
        batch_size=8,
        seq_length=4096,
        step_time_ms=500,
        mfu_percent=38,
        overlap_ratio=0.65,
        comm_time_ratio=0.22,
        precision="bf16",
    )
    
    simulator = WhatIfSimulator(state)
    result = simulator.run_all_scenarios()
    
    print(f"\n当前状态总结:")
    print(f"  硬件: {state.hardware_name} x {state.num_devices}")
    print(f"  并行: TP{state.tp_size}_PP{state.pp_size}_DP{state.dp_size}")
    print(f"  性能: {state.step_time_ms}ms/step, MFU={state.mfu_percent}%")
    
    print(f"\n模拟场景数: {len(result.scenarios)}")
    print(f"\nTop 5 优化方案:")
    for i, scenario in enumerate(result.scenarios[:5], 1):
        speedup_pct = (scenario.predicted_speedup - 1) * 100
        print(f"  {i}. {scenario.name}")
        print(f"     加速: {'+' if speedup_pct > 0 else ''}{speedup_pct:.1f}%, 置信度: {scenario.confidence:.0%}")
    
    if result.best_scenario:
        print(f"\n推荐方案: {result.best_scenario.name}")
        print(f"  预测加速: {result.best_scenario.predicted_speedup:.2f}x")
        if result.best_scenario.benefits:
            print(f"  主要优势:")
            for b in result.best_scenario.benefits[:2]:
                print(f"    - {b}")
    
    print(f"\n{result.summary}")


def main():
    """运行所有测试"""
    print("Phase 9: Roofline & What-if 功能测试")
    print("=" * 60)
    
    test_roofline_basics()
    test_operator_analysis()
    test_theoretical_mfu()
    test_whatif_parallel()
    test_whatif_hardware()
    test_whatif_optimization()
    test_full_simulation()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
