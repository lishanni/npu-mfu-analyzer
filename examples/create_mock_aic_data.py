"""
创建模拟的 AIC Metrics 数据用于测试验证

模拟 msprof op --aic-metrics 生成的目录结构和 CSV 文件
"""

import os
from pathlib import Path
import pandas as pd


def create_mock_aic_metrics(base_dir: str = "/tmp/mock_aic_profiling"):
    """
    创建模拟的 AIC metrics 数据

    目录结构:
    OPPROF_20240101_120000_XXX/
    ├── MatMulV2/
    │   └── 0/
    │       ├── ArithmeticUtilization_0.csv
    │       ├── L2Cache_0.csv
    │       ├── Memory_0.csv
    │       ├── PipeUtilization_0.csv
    │       └── OpBasicInfo_0.csv
    ├── Add/
    │   └── 0/
    │       └── ...
    └── LayerNorm/
        └── 0/
            └── ...
    """
    base_path = Path(base_dir)
    opprof_dir = base_path / "OPPROF_20240101_120000_XXX"

    # 示例算子及其模拟指标
    mock_operators = [
        {
            "name": "MatMulV2",
            "arithmetic": {
                "Cube_Utilization": [15.5],  # 计算瓶颈
                "Vector_Utilization": [35.2],
                "Scalar_Utilization": [8.5],
                "Total_Cycles": [1000000],
            },
            "l2_cache": {
                "L2_Hit_Rate": [85.0],
                "Read_Bandwidth_GBps": [120.5],
                "Write_Bandwidth_GBps": [45.2],
            },
            "memory": {
                "UB_Usage_Percent": [65.0],
                "L0_Usage_Percent": [45.0],
            },
            "pipeline": {
                "Pipeline_Utilization": [55.0],
                "Stall_Rate": [25.0],
            },
            "basic_info": {
                "Duration_us": [1250.5],
                "Op_Type": ["Compute"],
            },
        },
        {
            "name": "Add",
            "arithmetic": {
                "Cube_Utilization": [5.0],  # 严重计算瓶颈（简单操作）
                "Vector_Utilization": [45.0],
                "Scalar_Utilization": [25.0],
                "Total_Cycles": [500000],
            },
            "l2_cache": {
                "L2_Hit_Rate": [95.0],  # 良好的缓存命中率
                "Read_Bandwidth_GBps": [50.0],
                "Write_Bandwidth_GBps": [30.0],
            },
            "memory": {
                "UB_Usage_Percent": [25.0],
                "L0_Usage_Percent": [20.0],
            },
            "pipeline": {
                "Pipeline_Utilization": [35.0],  # 流水线利用率低
                "Stall_Rate": [55.0],  # 高停顿率
            },
            "basic_info": {
                "Duration_us": [85.5],
                "Op_Type": ["ElementWise"],
            },
        },
        {
            "name": "LayerNorm",
            "arithmetic": {
                "Cube_Utilization": [25.0],  # 中等利用率
                "Vector_Utilization": [55.0],
                "Scalar_Utilization": [15.0],
                "Total_Cycles": [800000],
            },
            "l2_cache": {
                "L2_Hit_Rate": [45.0],  # 内存瓶颈 - 缓存命中率低
                "Read_Bandwidth_GBps": [180.5],
                "Write_Bandwidth_GBps": [90.2],
            },
            "memory": {
                "UB_Usage_Percent": [85.0],  # 高 UB 使用率
                "L0_Usage_Percent": [75.0],
            },
            "pipeline": {
                "Pipeline_Utilization": [45.0],
                "Stall_Rate": [35.0],
            },
            "basic_info": {
                "Duration_us": [580.5],
                "Op_Type": ["Memory"],
            },
        },
    ]

    for op in mock_operators:
        op_dir = opprof_dir / op["name"] / "0"
        op_dir.mkdir(parents=True, exist_ok=True)

        # 创建 ArithmeticUtilization CSV
        pd.DataFrame(op["arithmetic"]).to_csv(
            op_dir / "ArithmeticUtilization_0.csv", index=False
        )

        # 创建 L2Cache CSV
        pd.DataFrame(op["l2_cache"]).to_csv(
            op_dir / "L2Cache_0.csv", index=False
        )

        # 创建 Memory CSV
        pd.DataFrame(op["memory"]).to_csv(
            op_dir / "MemoryUB_0.csv", index=False
        )

        # 创建 PipeUtilization CSV
        pd.DataFrame(op["pipeline"]).to_csv(
            op_dir / "PipeUtilization_0.csv", index=False
        )

        # 创建 OpBasicInfo CSV
        pd.DataFrame(op["basic_info"]).to_csv(
            op_dir / "OpBasicInfo_0.csv", index=False
        )

    print(f"✅ 创建模拟 AIC metrics 数据: {opprof_dir}")
    print(f"   - {len(mock_operators)} 个算子")
    print(f"   - MatMulV2: 计算瓶颈 (Cube: 15.5%)")
    print(f"   - Add: 流水线瓶颈 (Stall: 55.0%)")
    print(f"   - LayerNorm: 内存瓶颈 (L2 Hit: 45.0%, UB: 85.0%)")

    return str(opprof_dir)


if __name__ == "__main__":
    create_mock_aic_metrics()
