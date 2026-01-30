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
        
        Returns:
            DataFrame with columns: step, computing, communication, 
                                   communication_not_overlapped, overlapped, free, stage, bubble
        """
        info = self.detect()
        
        if info.data_type == "db":
            return self._get_step_trace_from_db(rank)
        else:
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
            logger.error(f"Error reading STEP_TRACE: {e}")
            return pd.DataFrame()
    
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
