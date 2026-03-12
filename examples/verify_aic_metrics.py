"""
AIC Metrics 分析功能验证脚本

演示如何使用扩展后的 npu-mfu-analyzer 分析 AIC metrics 数据
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader
from npu_mfu_analyzer.data_loader.aic_metrics import AICMetrics
from examples.create_mock_aic_data import create_mock_aic_metrics


def verify_aic_metrics_parsing():
    """验证 AIC metrics 解析功能"""
    print("=" * 60)
    print("验证 AIC Metrics 解析功能")
    print("=" * 60)

    # 1. 创建模拟数据
    print("\n[步骤 1] 创建模拟 AIC metrics 数据...")
    profiling_path = create_mock_aic_metrics()

    # 2. 初始化 ProfilingLoader
    print("\n[步骤 2] 初始化 ProfilingLoader...")
    loader = ProfilingLoader(profiling_path)
    print(f"✅ ProfilingLoader 初始化成功")

    # 3. 获取 AIC metrics
    print("\n[步骤 3] 获取 AIC metrics...")
    aic_metrics = loader.get_aic_metrics()
    print(f"✅ 加载了 {len(aic_metrics)} 个算子的 AIC metrics")

    # 4. 分析每个算子
    print("\n[步骤 4] 分析每个算子的硬件指标...")
    print("-" * 60)

    for op_name, metrics in aic_metrics.items():
        print(f"\n📊 {op_name}")
        print(f"   类型: {metrics.op_type}")
        print(f"   执行时间: {metrics.duration_us:.2f} μs")

        if metrics.arithmetic:
            print(f"   算术单元:")
            print(f"     - Cube 利用率: {metrics.arithmetic.cube_utilization:.1f}%")
            print(f"     - Vector 利用率: {metrics.arithmetic.vector_utilization:.1f}%")
            print(f"     - Scalar 利用率: {metrics.arithmetic.scalar_utilization:.1f}%")

        if metrics.memory:
            print(f"   内存:")
            print(f"     - L2 缓存命中率: {metrics.memory.l2_cache_hit_rate:.1f}%")
            print(f"     - UB 使用率: {metrics.memory.ub_usage:.1f}%")
            print(f"     - L0 使用率: {metrics.memory.l0_usage:.1f}%")

        if metrics.pipeline:
            print(f"   流水线:")
            print(f"     - 流水线利用率: {metrics.pipeline.pipe_utilization:.1f}%")
            print(f"     - 停顿率: {metrics.pipeline.stall_rate:.1f}%")

    # 5. 瓶颈分析
    print("\n" + "=" * 60)
    print("瓶颈分析总结")
    print("=" * 60)

    # 导入瓶颈判断常量
    from npu_mfu_analyzer.data_loader.aic_metrics import (
        CRITICAL_THRESHOLD,
        HIGH_THRESHOLD,
        BOTTLENECK_COMPUTE,
        BOTTLENECK_MEMORY,
        BOTTLENECK_PIPELINE,
    )

    for op_name, metrics in aic_metrics.items():
        cube_util = metrics.arithmetic.cube_utilization if metrics.arithmetic else 100.0
        l2_hit = metrics.memory.l2_cache_hit_rate if metrics.memory else 100.0
        stall = metrics.pipeline.stall_rate if metrics.pipeline else 0.0

        # 判断瓶颈
        if cube_util < CRITICAL_THRESHOLD:
            bottleneck = "计算瓶颈 (严重)"
            emoji = "🔴"
        elif l2_hit < CRITICAL_THRESHOLD:
            bottleneck = "内存瓶颈 (严重)"
            emoji = "🔴"
        elif stall > 50:
            bottleneck = "流水线瓶颈"
            emoji = "🟡"
        elif cube_util < HIGH_THRESHOLD:
            bottleneck = "计算瓶颈 (高)"
            emoji = "🟠"
        elif l2_hit < HIGH_THRESHOLD:
            bottleneck = "内存瓶颈 (高)"
            emoji = "🟠"
        else:
            bottleneck = "均衡"
            emoji = "🟢"

        print(f"\n{emoji} {op_name}: {bottleneck}")

    print("\n" + "=" * 60)
    print("✅ 验证完成！")
    print("=" * 60)
    print(f"\n模拟数据路径: {profiling_path}")
    print("您可以使用此路径测试其他功能，如 DetailedOperatorAgent 等")


async def verify_detailed_operator_agent():
    """验证 DetailedOperatorAgent 分析功能"""
    print("\n\n" + "=" * 60)
    print("验证 DetailedOperatorAgent")
    print("=" * 60)

    # 创建模拟数据
    profiling_path = create_mock_aic_data()

    # 注意: 实际使用需要配置 LLM，这里只演示数据流转
    print("\n⚠️  注意: DetailedOperatorAgent 需要 LLM 配置")
    print("   这里演示数据准备部分...")

    loader = ProfilingLoader(profiling_path)
    aic_metrics = loader.get_aic_metrics()

    # 模拟 Agent 分析流程
    print("\n[模拟分析流程]")
    for op_name, metrics in list(aic_metrics.items())[:2]:
        print(f"\n分析 {op_name}:")

        # 识别瓶颈
        cube_util = metrics.arithmetic.cube_utilization if metrics.arithmetic else 100
        l2_hit = metrics.memory.l2_cache_hit_rate if metrics.memory else 100
        stall = metrics.pipeline.stall_rate if metrics.pipeline else 0

        if cube_util < 20:
            bottleneck_type = "compute"
            severity = "critical"
        elif l2_hit < 60:
            bottleneck_type = "memory"
            severity = "high"
        elif stall > 40:
            bottleneck_type = "pipeline"
            severity = "high"
        else:
            bottleneck_type = "balanced"
            severity = "low"

        print(f"   - 瓶颈类型: {bottleneck_type}")
        print(f"   - 严重程度: {severity}")

        # 生成诊断
        if bottleneck_type == "compute":
            diagnosis = f"Cube 利用率仅 {cube_util:.1f}%，远低于理论峰值"
        elif bottleneck_type == "memory":
            diagnosis = f"L2 缓存命中率为 {l2_hit:.1f}%，数据局部性不佳"
        else:
            diagnosis = "各项指标较为均衡"

        print(f"   - 诊断: {diagnosis}")

    print("\n✅ DetailedOperatorAgent 数据流程验证完成")


if __name__ == "__main__":
    # 执行验证
    verify_aic_metrics_parsing()

    # 可选: 如果需要验证 Agent，取消下面的注释
    # asyncio.run(verify_detailed_operator_agent())
