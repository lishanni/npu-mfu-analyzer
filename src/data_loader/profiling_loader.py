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
        """统计 rank 数量"""
        import re
        ranks = set()
        
        for path in db_paths + json_paths:
            # 匹配 rank_0, rank_1 等
            match = re.search(r'rank[_-]?(\d+)', path, re.IGNORECASE)
            if match:
                ranks.add(int(match.group(1)))
            
            # 匹配 profiler_0.db, profiler_1.db 等
            match = re.search(r'profiler[_-]?(\d+)', path, re.IGNORECASE)
            if match:
                ranks.add(int(match.group(1)))
        
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
        """获取指定 rank 的 DB 路径"""
        info = self.detect()
        
        if not info.db_paths:
            return None
        
        if rank is not None:
            # 查找指定 rank 的 DB
            for path in info.db_paths:
                if f"rank_{rank}" in path or f"rank{rank}" in path or f"profiler_{rank}" in path:
                    return path
        
        # 返回第一个可用的
        return info.db_paths[0] if info.db_paths else None
    
    def _get_json_path(self, rank: Optional[int], file_type: str) -> Optional[str]:
        """获取指定 rank 的 JSON 路径"""
        info = self.detect()
        
        for path in info.json_paths:
            if file_type in path:
                if rank is None:
                    return path
                if f"rank_{rank}" in path or f"rank{rank}" in path:
                    return path
        
        # 返回第一个匹配的
        for path in info.json_paths:
            if file_type in path:
                return path
        
        return None
    
    def close(self):
        """关闭所有数据库连接"""
        for conn in self._db_connections.values():
            conn.close()
        self._db_connections.clear()
