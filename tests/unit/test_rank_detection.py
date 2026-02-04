"""
测试 rank/worker 识别功能

验证 ProfilingLoader 能正确识别各种命名模式的 rank/worker。
"""

import pytest
import tempfile
import json
from pathlib import Path


class TestRankDetection:
    """测试 rank/worker 识别"""

    def _create_trace_file(self, parent_dir: Path, name: str):
        """创建测试用的 trace_view.json 文件"""
        trace_file = parent_dir / name
        trace_file.write_text('[{"name": "test", "ts": 100, "dur": 10}]')
        return trace_file

    def test_count_ranks_with_rank_prefix(self, tmp_path):
        """测试识别 rank_ 前缀"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 创建多个 rank 的文件
        for i in range(4):
            rank_dir = tmp_path / f"rank_{i}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(rank_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 4

    def test_count_ranks_with_worker_dash(self, tmp_path):
        """测试识别 worker- 前缀（带连字符）"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 创建多个 worker 的文件
        for i in range(8):
            worker_dir = tmp_path / f"worker-{i}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(worker_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 8

    def test_count_ranks_with_worker_underscore(self, tmp_path):
        """测试识别 worker_ 前缀（带下划线）"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 创建多个 worker 的文件
        for i in range(3):
            worker_dir = tmp_path / f"worker_{i}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(worker_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 3

    def test_count_ranks_with_profiler_prefix(self, tmp_path):
        """测试识别 profiler_ 前缀"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 创建多个 profiler 的目录（使用 trace_view.json 模拟）
        for i in range(2):
            profiler_dir = tmp_path / f"profiler_{i}"
            profiler_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(profiler_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 2

    def test_count_ranks_complex_directory_structure(self, tmp_path):
        """测试识别复杂目录结构"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 模拟真实的复杂目录结构
        # profiler_zarr/ma-job-xxx-worker-0_yyy/ASCEND_PROFILER_OUTPUT/trace_view.json
        base_dir = tmp_path / "profiler_zarr"
        for i in range(4):
            worker_dir = base_dir / f"ma-job-abc123-worker-{i}_def456" / "ASCEND_PROFILER_OUTPUT"
            worker_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(worker_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 4

    def test_count_ranks_no_rank_returns_one(self, tmp_path):
        """测试没有 rank 信息时返回 1"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 只有一个通用的 trace_view.json
        self._create_trace_file(tmp_path, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 1

    def test_get_json_path_with_worker_dash(self, tmp_path):
        """测试获取 worker- 模式的 JSON 路径"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 创建多个 worker
        for i in range(3):
            worker_dir = tmp_path / f"worker-{i}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(worker_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))

        # 获取 worker-1 的路径
        path = loader._get_json_path(rank=1, file_type="trace_view")
        assert path is not None
        assert "worker-1" in path or "worker_1" in path

    def test_get_json_path_with_rank_prefix(self, tmp_path):
        """测试获取 rank_ 模式的 JSON 路径"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 创建多个 rank
        for i in range(3):
            rank_dir = tmp_path / f"rank_{i}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(rank_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))

        # 获取 rank_2 的路径
        path = loader._get_json_path(rank=2, file_type="trace_view")
        assert path is not None
        assert "rank_2" in path or "rank2" in path

    def test_mixed_naming_conventions(self, tmp_path):
        """测试混合命名约定"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 混合使用不同的命名方式
        rank_dir = tmp_path / "rank_0"
        rank_dir.mkdir(parents=True, exist_ok=True)
        self._create_trace_file(rank_dir, "trace_view.json")

        worker_dir = tmp_path / "worker-1"
        worker_dir.mkdir(parents=True, exist_ok=True)
        self._create_trace_file(worker_dir, "trace_view.json")

        profiler_dir = tmp_path / "profiler_2"
        profiler_dir.mkdir(parents=True, exist_ok=True)
        self._create_trace_file(profiler_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        # 应该识别出 3 个不同的 rank
        assert info.rank_count == 3

    def test_case_insensitive_matching(self, tmp_path):
        """测试大小写不敏感的匹配"""
        from src.data_loader.profiling_loader import ProfilingLoader

        # 使用大写命名
        for i in range(2):
            worker_dir = tmp_path / f"Worker-{i}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            self._create_trace_file(worker_dir, "trace_view.json")

        loader = ProfilingLoader(str(tmp_path))
        info = loader.detect()

        assert info.rank_count == 2
