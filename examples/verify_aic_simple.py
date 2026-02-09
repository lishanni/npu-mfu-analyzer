"""
AIC Metrics 功能演示 (简化版 - 不依赖 pandas)

直接演示数据结构和分析逻辑，无需运行完整工具
"""

import csv
from pathlib import Path


def create_simple_mock_data():
    """创建简化的模拟 AIC metrics CSV 文件"""
    base_dir = Path("/tmp/mock_aic_simple")
    opprof_dir = base_dir / "OPPROF_20240101_120000_XXX"

    # MatMulV2 - 计算瓶颈示例
    matmul_dir = opprof_dir / "MatMulV2" / "0"
    matmul_dir.mkdir(parents=True, exist_ok=True)

    # ArithmeticUtilization
    with open(matmul_dir / "ArithmeticUtilization_0.csv", "w") as f:
        f.write("Cube_Utilization,Vector_Utilization,Scalar_Utilization,Total_Cycles\n")
        f.write("15.5,35.2,8.5,1000000\n")

    # L2Cache
    with open(matmul_dir / "L2Cache_0.csv", "w") as f:
        f.write("L2_Hit_Rate,Read_Bandwidth_GBps,Write_Bandwidth_GBps\n")
        f.write("85.0,120.5,45.2\n")

    # PipeUtilization
    with open(matmul_dir / "PipeUtilization_0.csv", "w") as f:
        f.write("Pipeline_Utilization,Stall_Rate\n")
        f.write("55.0,25.0\n")

    # OpBasicInfo
    with open(matmul_dir / "OpBasicInfo_0.csv", "w") as f:
        f.write("Duration_us,Op_Type\n")
        f.write("1250.5,Compute\n")

    # LayerNorm - 内存瓶颈示例
    norm_dir = opprof_dir / "LayerNorm" / "0"
    norm_dir.mkdir(parents=True, exist_ok=True)

    with open(norm_dir / "ArithmeticUtilization_0.csv", "w") as f:
        f.write("Cube_Utilization,Vector_Utilization,Scalar_Utilization,Total_Cycles\n")
        f.write("25.0,55.0,15.0,800000\n")

    with open(norm_dir / "L2Cache_0.csv", "w") as f:
        f.write("L2_Hit_Rate,Read_Bandwidth_GBps,Write_Bandwidth_GBps\n")
        f.write("45.0,180.5,90.2\n")

    with open(norm_dir / "PipeUtilization_0.csv", "w") as f:
        f.write("Pipeline_Utilization,Stall_Rate\n")
        f.write("45.0,35.0\n")

    with open(norm_dir / "OpBasicInfo_0.csv", "w") as f:
        f.write("Duration_us,Op_Type\n")
        f.write("580.5,Memory\n")

    print(f"✅ 创建模拟数据: {opprof_dir}")
    return str(opprof_dir)


def parse_aic_csv(csv_path):
    """简化的 CSV 解析"""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                try:
                    data[key.lower().replace(' ', '_')] = float(value)
                except ValueError:
                    data[key.lower().replace(' ', '_')] = value
    return data


def analyze_operator_bottleneck(arithmetic, l2_cache, pipeline):
    """分析算子瓶颈"""
    cube = arithmetic.get('cube_utilization', 100)
    l2_hit = l2_cache.get('l2_hit_rate', 100)
    stall = pipeline.get('stall_rate', 0)

    # 瓶颈判断
    if cube < 20:
        return "compute", "critical", f"Cube 利用率极低 ({cube:.1f}%)"
    elif l2_hit < 50:
        return "memory", "critical", f"L2 缓存命中率极低 ({l2_hit:.1f}%)"
    elif stall > 50:
        return "pipeline", "high", f"流水线停顿率过高 ({stall:.1f}%)"
    elif cube < 40:
        return "compute", "high", f"Cube 利用率偏低 ({cube:.1f}%)"
    else:
        return "balanced", "low", "各项指标较为均衡"


def main():
    print("=" * 70)
    print("AIC Metrics 分析功能演示")
    print("=" * 70)

    # 创建模拟数据
    print("\n[步骤 1] 创建模拟 AIC metrics 数据...")
    profiling_path = create_simple_mock_data()

    # 模拟解析和分析
    print("\n[步骤 2] 分析算子性能...")

    operators = [
        ("MatMulV2", "compute"),
        ("LayerNorm", "memory"),
    ]

    print("\n" + "-" * 70)
    print("分析结果")
    print("-" * 70)

    for op_name, expected_bottleneck in operators:
        print(f"\n📊 {op_name}")

        # 模拟解析
        op_dir = Path(profiling_path) / op_name / "0"

        try:
            arithmetic = parse_aic_csv(op_dir / "ArithmeticUtilization_0.csv")
            l2_cache = parse_aic_csv(op_dir / "L2Cache_0.csv")
            pipeline = parse_aic_csv(op_dir / "PipeUtilization_0.csv")
            basic = parse_aic_csv(op_dir / "OpBasicInfo_0.csv")

            # 显示指标
            print(f"   执行时间: {basic.get('duration_us', 0):.1f} μs")
            print(f"   Cube 利用率: {arithmetic.get('cube_utilization', 0):.1f}%")
            print(f"   L2 缓存命中率: {l2_cache.get('l2_hit_rate', 0):.1f}%")
            print(f"   流水线利用率: {pipeline.get('pipeline_utilization', 0):.1f}%")
            print(f"   停顿率: {pipeline.get('stall_rate', 0):.1f}%")

            # 分析瓶颈
            bottleneck_type, severity, diagnosis = analyze_operator_bottleneck(
                arithmetic, l2_cache, pipeline
            )

            emoji = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟢"

            print(f"\n   {emoji} 瓶颈分析:")
            print(f"     类型: {bottleneck_type}")
            print(f"     严重度: {severity}")
            print(f"     诊断: {diagnosis}")

        except Exception as e:
            print(f"   ❌ 解析失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 演示完成")
    print("=" * 70)

    print(f"\n模拟数据位置: {profiling_path}")
    print("\n完整功能需要:")
    print("  1. 安装 pandas: pip install pandas")
    print("  2. 配置 LLM 后端")
    print("  3. 运行: python -m src.cli.main analyze --profiling-path <path>")


if __name__ == "__main__":
    main()
