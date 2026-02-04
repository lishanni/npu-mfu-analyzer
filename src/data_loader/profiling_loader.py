"""
Profiling 数据加载器

自动检测数据格式（DB/JSON），提供统一的数据访问接口。
"""

import os
import glob
import logging
import sqlite3
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_loader.stream_parser import StreamParser, TimelineSummarizer, extract_overlap_events

logger = logging.getLogger(__name__)


@dataclass
class ProfilingInfo:
    """Profiling 数据信息"""
    path: str
    data_type: str  # "db" or "json"
    framework: str  # "pytorch", "mindspore", "msprof"
    rank_count: int
    has_timeline: bool
    has_memory: bool
    has_communication: bool
    db_paths: List[str]
    json_paths: List[str]


class ProfilingLoader:
    """
    Profiling 数据加载器
    
    自动检测数据格式，提供统一的数据访问接口。
    优先使用 DB 格式（支持索引查询，更高效）。
    
    Usage:
        loader = ProfilingLoader("/path/to/profiling")
        info = loader.detect()
        timeline_summary = loader.get_timeline_summary()
        step_trace = loader.get_step_trace()
    """
    
    # 支持的 DB 文件模式
    DB_PATTERNS = [
        "**/ascend_pytorch_profiler_*.db",
        "**/analysis.db",
        "**/msprof_*.db",
        "**/cluster_analysis.db",
    ]
    
    # 支持的 JSON 文件模式
    JSON_PATTERNS = [
        "**/trace_view.json",
        "**/msprof_*.json",
        "**/communication.json",
    ]

    # Step Trace CSV（msprof-analyze 等工具生成的交付件，DB 无 STEP_TRACE 时降级使用）
    STEP_TRACE_CSV = "step_trace_time.csv"

    # step_trace_time.csv 列名（小写后）到内部列名的映射
    _STEP_TRACE_CSV_COLUMN_MAP = {
        "step": "step",
        "computing": "computing",
        "communication": "communication",
        "communication(not overlapped)": "communication_not_overlapped",
        "overlapped": "overlapped",
        "free": "free",
        "stage": "stage",
        "bubble": "bubble",
    }
    
    def __init__(self, profiling_path: str):
        """
        Args:
            profiling_path: Profiling 数据目录路径
        """
        self.profiling_path = Path(profiling_path)
        self._info: Optional[ProfilingInfo] = None
        self._db_connections: Dict[str, sqlite3.Connection] = {}
    
    def detect(self) -> ProfilingInfo:
        """
        检测 Profiling 数据类型和结构
        
        Returns:
            ProfilingInfo: 数据信息
        """
        if self._info is not None:
            return self._info
        
        db_paths = []
        json_paths = []
        
        # 搜索 DB 文件
        for pattern in self.DB_PATTERNS:
            db_paths.extend(glob.glob(str(self.profiling_path / pattern), recursive=True))
        
        # 搜索 JSON 文件
        for pattern in self.JSON_PATTERNS:
            json_paths.extend(glob.glob(str(self.profiling_path / pattern), recursive=True))
        
        # 确定数据类型
        if db_paths:
            data_type = "db"
        elif json_paths:
            data_type = "json"
        else:
            data_type = "unknown"
        
        # 确定框架类型
        framework = self._detect_framework(db_paths, json_paths)
        
        # 统计 rank 数量
        rank_count = self._count_ranks(db_paths, json_paths)
        
        # 检测数据可用性
        has_timeline = any("trace_view" in p for p in json_paths) or \
                      any("pytorch_profiler" in p for p in db_paths)
        has_memory = any("memory" in p.lower() for p in json_paths + db_paths)
        has_communication = any("communication" in p.lower() for p in json_paths) or \
                           any("cluster_analysis" in p for p in db_paths)
        
        self._info = ProfilingInfo(
            path=str(self.profiling_path),
            data_type=data_type,
            framework=framework,
            rank_count=rank_count,
            has_timeline=has_timeline,
            has_memory=has_memory,
            has_communication=has_communication,
            db_paths=sorted(db_paths),
            json_paths=sorted(json_paths),
        )
        
        logger.info(f"Detected profiling data: {self._info}")
        return self._info
    
    def _detect_framework(self, db_paths: List[str], json_paths: List[str]) -> str:
        """检测框架类型"""
        all_paths = " ".join(db_paths + json_paths).lower()
        
        if "pytorch" in all_paths or "ascend_pytorch" in all_paths:
            return "pytorch"
        elif "mindspore" in all_paths:
            return "mindspore"
        elif "msprof" in all_paths:
            return "msprof"
        return "unknown"
    
    def _count_ranks(self, db_paths: List[str], json_paths: List[str]) -> int:
        """
        统计 rank/worker 数量

        支持多种命名模式：
        - rank_0, rank_1
        - worker-0, worker-1
        - profiler_0.db, profiler_1.db
        - card_0, device_0
        - ma-job-xxx-worker-0_yyy (复杂目录结构)
        以及任何包含数字序号的路径模式
        """
        import re
        ranks = set()

        # 常见的 rank/worker 前缀模式（使用单词边界确保精确匹配）
        # 模式解释：
        # - (?:...) 非捕获组
        # - [_-]? 可选的下划线或连字符
        # - (\d+) 捕获数字（rank 编号）
        # - (?:_|/|\.|:) 数字后的分隔符（确保不是更大数字的一部分）
        patterns = [
            # 匹配 /worker-0/, /rank_1/, profiler_2.db 等
            r'/(?:rank|worker|profiler|card|device|node)[_-]?(\d+)(?=_|/|\.|:)',
            # 匹配路径开头的 rank_0/, worker-1/ 等
            r'^(?:rank|worker|profiler|card|device|node)[_-]?(\d+)(?=_|/|\.|:)',
            # 匹配复杂结构如 ma-job-xxx-worker-0_yyy
            r'(?:rank|worker|profiler|card|device|node)[_-]?(\d+)(?=_|/|\.|:|$)',
        ]

        for path in db_paths + json_paths:
            # 首先尝试匹配已知前缀（精确匹配）
            for pattern in patterns:
                matches = re.finditer(pattern, path, re.IGNORECASE)
                for match in matches:
                    rank_num = int(match.group(1))
                    ranks.add(rank_num)

            # 如果没有匹配到已知前缀，尝试从目录结构推断
            # 例如: ASCEND_PROFILER_OUTPUT/worker-0/trace_view.json
            #      profiler_zarr/ma-job-xxx-worker-0_yyy/trace_view.json
            path_parts = Path(path).parts

            # 查找包含数字序号的目录部分
            for part in path_parts:
                # 匹配 worker-0, worker-1 等模式
                match = re.search(r'(^|/)(?:worker|rank)[_-]?(\d+)(?:/|_|$)', part, re.IGNORECASE)
                if match:
                    ranks.add(int(match.group(2)))

                # 匹配其他可能的编号模式（如数字结尾的目录名）
                # 例如: worker0, rank1, node2 等
                match = re.search(r'(\d+)$', part)
                if match and len(part) < 20:  # 避免匹配到过长的哈希字符串
                    num = int(match.group(1))
                    # 排除一些明显不是 rank 的目录名模式
                    if (0 <= num <= 1024 and
                        not part.startswith('pytest') and
                        not part.startswith('test_') and
                        not part.startswith('test_count_ranks') and
                        'tmp' not in part.lower() and
                        'temp' not in part.lower()):
                        ranks.add(num)

        logger.debug(f"Detected ranks: {sorted(ranks)}")
        return len(ranks) if ranks else 1
    
    def get_timeline_summary(self, rank: Optional[int] = None) -> Dict[str, Any]:
        """
        获取 Timeline 摘要
        
        Args:
            rank: 指定 rank，None 表示使用第一个可用的
            
        Returns:
            Timeline 摘要统计
        """
        info = self.detect()
        
        if info.data_type == "db":
            return self._get_timeline_summary_from_db(rank)
        else:
            return self._get_timeline_summary_from_json(rank)
    
    def _get_timeline_summary_from_db(self, rank: Optional[int] = None) -> Dict[str, Any]:
        """从 DB 获取 Timeline 摘要"""
        db_path = self._get_db_path(rank)
        if not db_path:
            return {}
        
        try:
            conn = sqlite3.connect(db_path)
            
            # 尝试从 STEP_TRACE 表获取数据
            try:
                df = pd.read_sql_query("""
                    SELECT step, computing, communication, 
                           communication_not_overlapped, overlapped, free
                    FROM STEP_TRACE
                    LIMIT 100
                """, conn)
                
                if not df.empty:
                    return {
                        "source": "db",
                        "step_count": len(df),
                        "avg_computing": df["computing"].mean(),
                        "avg_communication": df["communication"].mean(),
                        "avg_comm_not_overlap": df.get("communication_not_overlapped", pd.Series([0])).mean(),
                        "avg_overlapped": df.get("overlapped", pd.Series([0])).mean(),
                        "avg_free": df["free"].mean(),
                        "raw_data": df.to_dict("records")[:10],  # 只保留前10条
                    }
            except Exception as e:
                logger.debug(f"STEP_TRACE not found: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error reading DB {db_path}: {e}")
        
        return {}
    
    def _get_timeline_summary_from_json(self, rank: Optional[int] = None) -> Dict[str, Any]:
        """从 JSON 获取 Timeline 摘要（流式解析）"""
        json_path = self._get_json_path(rank, "trace_view")
        if not json_path:
            return {}
        
        return TimelineSummarizer.summarize_file(json_path)
    
    def get_step_trace(self, rank: Optional[int] = None) -> pd.DataFrame:
        """
        获取 Step Trace 数据
        
        优先从 DB 的 STEP_TRACE 表读取；若表不存在或为空，则降级从 step_trace_time.csv 读取。
        
        Returns:
            DataFrame with columns: step, computing, communication,
                                   communication_not_overlapped, overlapped, free, stage, bubble
        """
        info = self.detect()

        if info.data_type == "db":
            df = self._get_step_trace_from_db(rank)
            if not df.empty:
                return df
            # 降级：DB 无 STEP_TRACE 时从 step_trace_time.csv 读
            df = self._get_step_trace_from_csv(rank)
            if not df.empty:
                logger.info("STEP_TRACE not in DB, using step_trace_time.csv as fallback")
            return df
        return pd.DataFrame()  # JSON 格式需要从 trace_view 解析

    def _get_step_trace_from_db(self, rank: Optional[int] = None) -> pd.DataFrame:
        """从 DB 获取 Step Trace"""
        db_path = self._get_db_path(rank)
        if not db_path:
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM STEP_TRACE", conn)
            conn.close()
            return df
        except Exception as e:
            logger.debug(f"STEP_TRACE not in DB: {e}")
            return pd.DataFrame()

    def _get_step_trace_from_csv(self, rank: Optional[int] = None) -> pd.DataFrame:
        """
        从 step_trace_time.csv 读取 Step Trace（降级数据源）。
        msprof-analyze 等工具会生成该 CSV，列名如：Step, Computing, Communication(Not Overlapped), Overlapped, Free, Stage, Bubble 等。
        """
        candidates = glob.glob(str(self.profiling_path / "**" / self.STEP_TRACE_CSV), recursive=True)
        if not candidates:
            return pd.DataFrame()

        # 若指定 rank，优先选含该 rank 的路径（如 ASCEND_PROFILER_OUTPUT/step_trace_time.csv 通常为单卡）
        csv_path = candidates[0]
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"Failed to read {csv_path}: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # 将 CSV 列名统一为小写并去掉多余空格，再映射到内部列名
        df = df.rename(columns=lambda c: str(c).strip().lower() if isinstance(c, str) else c)

        # 兼容多种列名写法：Communication(Not Overlapped) / communication(not overlapped)
        rename_map = {}
        for col in df.columns:
            key = col.strip().lower()
            if key in self._STEP_TRACE_CSV_COLUMN_MAP:
                rename_map[col] = self._STEP_TRACE_CSV_COLUMN_MAP[key]
            elif "communication" in key and "not overlapped" in key and "exclude" not in key:
                rename_map[col] = "communication_not_overlapped"
            elif "communication" in key and "exclude" in key:
                pass  # 可选列，不映射
            elif key == "device_id":
                pass  # 保留，用于多卡过滤
            elif key == "preparing":
                pass  # 可选列

        df = df.rename(columns=rename_map)

        # 若 CSV 有 device_id，且指定了 rank，则过滤
        if rank is not None and "device_id" in df.columns:
            df = df[df["device_id"] == rank]

        return df
    
    def get_overlap_events(self, rank: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        获取 Overlap 分析相关事件
        
        Returns:
            {
                "compute": [...],
                "comm_not_overlap": [...],
                "free": [...],
                "hccl": [...]
            }
        """
        json_path = self._get_json_path(rank, "trace_view")
        if json_path:
            return extract_overlap_events(json_path)
        return {"compute": [], "comm_not_overlap": [], "free": [], "hccl": []}
    
    def get_communication_data(self, rank: Optional[int] = None) -> pd.DataFrame:
        """获取通信数据"""
        info = self.detect()

        if info.data_type == "db":
            db_path = self._get_db_path(rank)
            if db_path:
                try:
                    conn = sqlite3.connect(db_path)
                    # 尝试多个可能的表名
                    for table in ["HCCL", "COMMUNICATION", "hccl_op"]:
                        try:
                            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                            if not df.empty:
                                conn.close()
                                return df
                        except:
                            continue
                    conn.close()
                except Exception as e:
                    logger.error(f"Error reading communication data: {e}")

        return pd.DataFrame()

    def get_top_kernels(self, rank: Optional[int] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        获取 Top N 耗时 Kernel 算子

        优先从 DB 的 TASK 表查询，其次从 op_statistic.csv 或 kernel_details.csv 读取。

        Args:
            rank: 指定 rank，None 表示使用第一个可用的
            top_n: 返回的算子数量

        Returns:
            [{"name": "kernel_name", "dur": 123.45, "cat": "Kernel"}, ...]
        """
        info = self.detect()
        kernels = []

        # 优先从 DB 获取
        db_path = None
        if info.data_type == "db" and info.db_paths:
            db_path = self._get_db_path(rank)

        # 如果没有找到 DB，尝试查找任意 .db 文件
        if db_path is None:
            db_candidates = glob.glob(str(self.profiling_path / "**/*.db"), recursive=True)
            if db_candidates:
                db_path = db_candidates[0]

        if db_path:
            try:
                from src.data_loader.db_query import DBQuery

                db = DBQuery(db_path)
                df = db.query_kernel_ops(top_n=top_n)
                db.close()

                if not df.empty:
                    for _, row in df.iterrows():
                        kernels.append({
                            "name": row.get("name", "unknown"),
                            "dur": float(row.get("dur", 0)),
                            "cat": row.get("cat", "Kernel"),
                        })
                    return kernels
            except Exception as e:
                logger.debug(f"Failed to get kernels from DB: {e}")

        # 尝试从 CSV 获取
        kernels = self._get_kernels_from_csv(rank, top_n)
        if kernels:
            return kernels

        # 最后从 trace_view.json 中过滤 Kernel 事件
        kernels = self._get_kernels_from_timeline(rank, top_n)

        return kernels

    def _get_kernels_from_csv(self, rank: Optional[int], top_n: int) -> List[Dict[str, Any]]:
        """从 CSV 文件获取 Top Kernel 算子"""
        csv_files = [
            "op_statistic.csv",
            "kernel_details.csv",
        ]

        for csv_file in csv_files:
            candidates = glob.glob(str(self.profiling_path / "**" / csv_file), recursive=True)
            if not candidates:
                continue

            try:
                df = pd.read_csv(candidates[0])

                # 标准化列名（去除空格并转小写）
                original_columns = list(df.columns)
                df.columns = [str(c).strip().lower() for c in df.columns]

                # 查找耗时相关列（支持更多变体）
                dur_col = None
                for col in ["duration", "dur", "time", "total_time", "avg_duration", "total_duration"]:
                    if col in df.columns:
                        dur_col = col
                        break

                if dur_col is None:
                    logger.debug(f"No duration column found in {csv_file}, columns: {df.columns}")
                    continue

                # 查找名称列（支持更多变体）
                name_col = None
                for col in ["name", "op_name", "kernel_name", "operator_name"]:
                    if col in df.columns:
                        name_col = col
                        break

                if name_col is None:
                    logger.debug(f"No name column found in {csv_file}, columns: {df.columns}")
                    continue

                # 按耗时排序并取 Top N
                df_sorted = df.sort_values(by=dur_col, ascending=False).head(top_n)

                kernels = []
                for _, row in df_sorted.iterrows():
                    kernels.append({
                        "name": str(row[name_col]),
                        "dur": float(row[dur_col]),
                        "cat": "Kernel",
                    })

                if kernels:
                    logger.info(f"Loaded {len(kernels)} kernels from {csv_file}")
                    return kernels

            except Exception as e:
                logger.debug(f"Failed to read {csv_file}: {e}")

        return []

    def _get_kernels_from_timeline(self, rank: Optional[int], top_n: int) -> List[Dict[str, Any]]:
        """从 trace_view.json 中过滤 Kernel 事件"""
        from decimal import Decimal

        try:
            json_path = self._get_json_path(rank, "trace_view")
            if not json_path:
                return []

            kernels = []
            for event in StreamParser(json_path).iter_events(show_progress=False):
                cat = event.get("cat", "")
                if cat == "Kernel" or cat.lower() == "kernel":
                    dur = event.get("dur", 0)
                    # 统一处理各种类型（包括 Decimal）
                    if isinstance(dur, str):
                        try:
                            dur = float(dur)
                        except ValueError:
                            dur = 0.0
                    elif isinstance(dur, Decimal):
                        dur = float(dur)
                    elif isinstance(dur, (int, float)):
                        dur = float(dur)
                    else:
                        dur = 0.0

                    kernels.append({
                        "name": event.get("name", "unknown"),
                        "dur": dur,
                        "cat": "Kernel",
                    })

            # 按耗时排序并取 Top N
            kernels.sort(key=lambda x: x["dur"], reverse=True)
            return kernels[:top_n]

        except Exception as e:
            logger.debug(f"Failed to get kernels from timeline: {e}")
            return []
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        info = self.detect()
        
        # 尝试从 info.json 获取
        info_files = glob.glob(str(self.profiling_path / "**/info.json*"), recursive=True)
        
        for info_file in info_files:
            try:
                import json
                with open(info_file, "r") as f:
                    data = json.load(f)
                    if "aicore_count" in str(data) or "device" in str(data):
                        return data
            except:
                continue
        
        return {}
    
    def _get_db_path(self, rank: Optional[int] = None) -> Optional[str]:
        """
        获取指定 rank 的 DB 路径

        支持多种命名模式：rank_0, worker-0, profiler_0, card_0 等
        """
        import re
        info = self.detect()

        if not info.db_paths:
            return None

        if rank is not None:
            # 多种匹配模式
            patterns = [
                f'rank[_-]?{rank}\\b',
                f'worker[_-]?{rank}\\b',
                f'profiler[_-]?{rank}\\b',
                f'card[_-]?{rank}\\b',
                f'device[_-]?{rank}\\b',
            ]

            for path in info.db_paths:
                for pattern in patterns:
                    if re.search(pattern, path, re.IGNORECASE):
                        return path

        # 返回第一个可用的
        return info.db_paths[0] if info.db_paths else None
    
    def _get_json_path(self, rank: Optional[int], file_type: str) -> Optional[str]:
        """
        获取指定 rank 的 JSON 路径

        支持多种命名模式：rank_0, worker-0, profiler_0 等
        """
        import re
        info = self.detect()

        # 首先尝试精确匹配指定的 rank
        if rank is not None:
            patterns = [
                f'rank[_-]?{rank}\\b',
                f'worker[_-]?{rank}\\b',
                f'profiler[_-]?{rank}\\b',
                f'card[_-]?{rank}\\b',
                f'device[_-]?{rank}\\b',
            ]

            for path in info.json_paths:
                if file_type in path:
                    for pattern in patterns:
                        if re.search(pattern, path, re.IGNORECASE):
                            return path

        # 返回第一个匹配的（rank=None 或未找到指定 rank）
        for path in info.json_paths:
            if file_type in path:
                return path

        return None
    
    def close(self):
        """关闭所有数据库连接"""
        for conn in self._db_connections.values():
            conn.close()
        self._db_connections.clear()
