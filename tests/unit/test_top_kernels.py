"""
测试 Top Kernel 算子获取功能

验证 ProfilingLoader 能正确获取真正的 Kernel 算子，而不是 Python 栈帧。
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path


class TestTopKernels:
    """测试 Top Kernel 算子获取"""

    def test_get_kernels_from_db(self, tmp_path):
        """测试从 DB 获取 Top Kernel 算子"""
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader

        # 创建测试数据库
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 创建 TASK 表
        cursor.execute("""
            CREATE TABLE TASK (
                id INTEGER,
                name TEXT,
                dur REAL,
                ts REAL,
                stream_id INTEGER,
                cat TEXT
            )
        """)

        # 插入测试数据（包含 Kernel 和其他类型）
        test_data = [
            (1, "MatMul", 50000.0, 1000.0, 0, "Kernel"),
            (2, "Add", 30000.0, 2000.0, 0, "Kernel"),
            (3, "Python::frame", 10000.0, 3000.0, 0, "Python"),
            (4, "Conv2D", 80000.0, 4000.0, 0, "Kernel"),
            (5, "ReduceSum", 20000.0, 5000.0, 0, "Kernel"),
        ]
        cursor.executemany("INSERT INTO TASK VALUES (?, ?, ?, ?, ?, ?)", test_data)
        conn.commit()
        conn.close()

        loader = ProfilingLoader(str(tmp_path))
        kernels = loader.get_top_kernels(top_n=3)

        # 应该只返回 Kernel 类型的算子，按耗时排序
        assert len(kernels) == 3
        assert kernels[0]["name"] == "Conv2D"
        assert kernels[0]["dur"] == 80000.0
        assert kernels[1]["name"] == "MatMul"
        assert kernels[1]["dur"] == 50000.0
        assert kernels[2]["name"] == "Add"
        assert kernels[2]["dur"] == 30000.0

    def test_get_kernels_from_op_statistic_csv(self, tmp_path):
        """测试从 op_statistic.csv 获取 Top Kernel 算子"""
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader

        # 创建测试 CSV 文件
        import pandas as pd
        csv_path = tmp_path / "op_statistic.csv"

        df = pd.DataFrame({
            "Name": ["MatMul", "Add", "Conv2D", "ReduceSum"],
            "Duration": [50000.0, 30000.0, 80000.0, 20000.0],
            "Type": ["Kernel", "Kernel", "Kernel", "Kernel"],
        })
        df.to_csv(csv_path, index=False)

        loader = ProfilingLoader(str(tmp_path))
        kernels = loader.get_top_kernels(top_n=3)

        assert len(kernels) == 3
        assert kernels[0]["name"] == "Conv2D"
        assert kernels[0]["dur"] == 80000.0

    def test_get_kernels_from_kernel_details_csv(self, tmp_path):
        """测试从 kernel_details.csv 获取 Top Kernel 算子"""
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader

        # 创建测试 CSV 文件
        import pandas as pd
        csv_path = tmp_path / "kernel_details.csv"

        df = pd.DataFrame({
            "Kernel_Name": ["MatMul", "Add", "Conv2D"],
            "Total_Duration": [50000.0, 30000.0, 80000.0],
        })
        df.to_csv(csv_path, index=False)

        loader = ProfilingLoader(str(tmp_path))
        kernels = loader.get_top_kernels(top_n=2)

        assert len(kernels) == 2
        assert kernels[0]["name"] == "Conv2D"
        assert kernels[0]["dur"] == 80000.0

    def test_get_kernels_from_timeline_filters_python(self, tmp_path):
        """测试从 timeline 过滤 Kernel，排除 Python 栈帧"""
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader

        # 创建测试 trace_view.json
        trace_file = tmp_path / "trace_view.json"

        events = [
            # Python 栈帧（应该被排除）
            {"name": "Python::PythonObserved::ProfilerStep#11", "cat": "Python", "dur": 58187.60, "ts": 1000},
            {"name": "megatron/training/training.py(577):train_step", "cat": "Python", "dur": 58164.99, "ts": 2000},
            # Kernel 算子（应该被包含）
            {"name": "MatMul", "cat": "Kernel", "dur": 50000.0, "ts": 3000},
            {"name": "Add", "cat": "Kernel", "dur": 30000.0, "ts": 4000},
            {"name": "Conv2D", "cat": "Kernel", "dur": 80000.0, "ts": 5000},
        ]
        trace_file.write_text(json.dumps(events))

        loader = ProfilingLoader(str(tmp_path))
        kernels = loader.get_top_kernels(top_n=3)

        # 应该只返回 Kernel 类型的算子
        assert len(kernels) == 3
        assert all(k["cat"] == "Kernel" for k in kernels)
        # 按耗时排序
        assert kernels[0]["name"] == "Conv2D"
        assert kernels[0]["dur"] == 80000.0
        assert kernels[1]["name"] == "MatMul"
        assert kernels[1]["dur"] == 50000.0

    def test_get_kernels_empty_data(self, tmp_path):
        """测试空数据返回空列表"""
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader

        loader = ProfilingLoader(str(tmp_path))
        kernels = loader.get_top_kernels()

        assert kernels == []

    def test_get_kernels_with_various_csv_formats(self, tmp_path):
        """测试各种 CSV 格式的兼容性"""
        from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader

        # 测试不同的列名格式
        test_cases = [
            # (列名映射, 数据)
            ({"name": "Op_Name", "dur": "Avg_Duration"}, [("MatMul", 50000.0), ("Add", 30000.0)]),
            ({"name": "kernel_name", "dur": "time"}, [("Conv2D", 80000.0), ("ReduceSum", 20000.0)]),
        ]

        for i, (columns, data) in enumerate(test_cases):
            csv_dir = tmp_path / f"test_{i}"
            csv_dir.mkdir()
            csv_path = csv_dir / "op_statistic.csv"

            import pandas as pd
            df = pd.DataFrame(data, columns=list(columns.values()))
            df.to_csv(csv_path, index=False)

            loader = ProfilingLoader(str(csv_dir))
            kernels = loader.get_top_kernels(top_n=2)

            assert len(kernels) == 2
            assert kernels[0]["dur"] == max(d for _, d in data)
