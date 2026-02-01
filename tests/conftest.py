"""
pytest 配置文件

提供全局 fixtures 和测试配置
"""

import sys
from pathlib import Path

import pytest

# 确保 src 目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 全局 Fixtures
# ============================================================

@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return PROJECT_ROOT / "tests" / "data"


@pytest.fixture(scope="session")
def sample_profiling_dir():
    """示例 profiling 数据目录（如果存在）"""
    # 查找可用的 profiling 数据
    candidates = [
        PROJECT_ROOT.parent / "msprof_3813_20260124100335956_ascend_pt-1",
        Path("/tmp/profiling_data"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ============================================================
# Mock Fixtures
# ============================================================

@pytest.fixture
def mock_timeline_events():
    """模拟 Timeline 事件数据"""
    return [
        {"name": "MatMul", "cat": "Computing", "ts": 0, "dur": 100, "pid": 0, "tid": 0},
        {"name": "AllReduce", "cat": "Communication", "ts": 100, "dur": 50, "pid": 0, "tid": 1},
        {"name": "LayerNorm", "cat": "Computing", "ts": 150, "dur": 30, "pid": 0, "tid": 0},
    ]


@pytest.fixture
def mock_operator_data():
    """模拟算子数据"""
    import pandas as pd
    return pd.DataFrame({
        "op_name": ["MatMul", "Conv2D", "LayerNorm", "Softmax"],
        "op_type": ["MatMul", "Conv2D", "LayerNorm", "Softmax"],
        "duration_us": [1000, 500, 200, 100],
        "input_shapes": ["[1,4096,4096]", "[1,3,224,224]", "[1,4096]", "[1,1024,1024]"],
    })


@pytest.fixture
def mock_communication_events():
    """模拟通信事件"""
    return [
        {"name": "AllReduce", "group": "tp_group", "dur": 5000, "data_size": 1024*1024*100},
        {"name": "ReduceScatter", "group": "dp_group", "dur": 3000, "data_size": 1024*1024*50},
        {"name": "AllGather", "group": "dp_group", "dur": 2000, "data_size": 1024*1024*50},
    ]


@pytest.fixture
def mock_step_trace_data():
    """模拟 step trace 数据"""
    import pandas as pd
    return pd.DataFrame({
        "step": [0, 1, 2, 3, 4],
        "iteration_time": [1000.0, 1010.0, 995.0, 1005.0, 1000.0],
        "computing_time": [800.0, 805.0, 798.0, 802.0, 800.0],
        "communication_time": [150.0, 155.0, 148.0, 152.0, 150.0],
        "free_time": [50.0, 50.0, 49.0, 51.0, 50.0],
    })


# ============================================================
# LLM Fixtures
# ============================================================

@pytest.fixture
def mock_llm_backend():
    """模拟 LLM 后端"""
    from src.llm import MockBackend
    return MockBackend()


# ============================================================
# Hardware Fixtures
# ============================================================

@pytest.fixture
def atlas_a2_spec():
    """Atlas A2 硬件规格"""
    from src.hardware import get_registry
    registry = get_registry()
    return registry.get_spec("Atlas A2", "280T")


# ============================================================
# pytest 配置
# ============================================================

def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama"
    )
