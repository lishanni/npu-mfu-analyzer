"""
数据库查询封装

支持从 SQLite DB 文件（ascend_pytorch_profiler_*.db, analysis.db）查询数据。
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DBQuery:
    """
    SQLite 数据库查询封装
    
    支持查询 msprof 生成的各种 DB 文件。
    
    Example:
        >>> db = DBQuery("ascend_pytorch_profiler_0.db")
        >>> df = db.query_table("STEP_TRACE")
        >>> print(df.columns)
    """
    
    # 常用表名
    TABLE_STEP_TRACE = "STEP_TRACE"
    TABLE_STEP_TIME = "STEP_TIME"
    TABLE_CANN_API = "CANN_API"
    TABLE_TASK = "TASK"
    TABLE_HCCL = "HCCL"
    TABLE_PYTORCH_API = "PYTORCH_API"
    TABLE_OP_MEMORY = "OP_MEMORY"
    TABLE_NPU_MEMORY = "NPU_MEMORY"
    
    def __init__(self, db_path: str):
        """
        初始化数据库连接
        
        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = Path(db_path)
        self._validate_db()
        self._engine: Optional[Engine] = None
    
    def _validate_db(self) -> None:
        """验证数据库文件"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        if not self.db_path.suffix == ".db":
            logger.warning(f"文件扩展名不是 .db: {self.db_path}")
    
    @property
    def engine(self) -> Engine:
        """获取数据库引擎（懒加载）"""
        if self._engine is None:
            self._engine = create_engine(f"sqlite:///{self.db_path}")
            logger.debug(f"已连接数据库: {self.db_path}")
        return self._engine
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.debug(f"已关闭数据库: {self.db_path}")
    
    @contextmanager
    def connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def get_tables(self) -> List[str]:
        """获取数据库中的所有表名"""
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """获取表的所有列名"""
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        return [col["name"] for col in columns]
    
    def query_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        查询表数据
        
        Args:
            table_name: 表名
            columns: 要查询的列，None 表示所有列
            where: WHERE 子句（不包含 WHERE 关键字）
            limit: 限制返回行数
            
        Returns:
            查询结果 DataFrame
        """
        # 构建 SQL
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table_name}"
        
        if where:
            sql += f" WHERE {where}"
        if limit:
            sql += f" LIMIT {limit}"
        
        logger.debug(f"执行查询: {sql}")
        
        return pd.read_sql(sql, self.engine)
    
    def query_sql(self, sql: str) -> pd.DataFrame:
        """
        执行自定义 SQL 查询
        
        Args:
            sql: SQL 查询语句
            
        Returns:
            查询结果 DataFrame
        """
        logger.debug(f"执行查询: {sql}")
        return pd.read_sql(sql, self.engine)
    
    def query_step_trace(
        self,
        step: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        查询 STEP_TRACE 表（包含 Overlap 数据）
        
        Args:
            step: 指定 Step ID，None 表示所有 Step
            columns: 要查询的列
            
        Returns:
            STEP_TRACE 数据
        """
        if columns is None:
            columns = [
                "step", "computing", "communication",
                "communication_not_overlapped", "overlapped",
                "free", "stage", "bubble"
            ]
        
        where = f"step = {step}" if step is not None else None
        
        return self.query_table(self.TABLE_STEP_TRACE, columns=columns, where=where)
    
    def query_step_time(self) -> pd.DataFrame:
        """查询 STEP_TIME 表"""
        return self.query_table(
            self.TABLE_STEP_TIME,
            columns=["id", "startNs", "endNs"]
        )
    
    def query_hccl_ops(self) -> pd.DataFrame:
        """查询 HCCL 通信算子"""
        return self.query_table(self.TABLE_HCCL)
    
    def query_kernel_ops(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        查询 Kernel 算子（按耗时排序）
        
        Args:
            top_n: 返回 Top N 耗时算子
        """
        sql = f"""
            SELECT name, dur, ts, stream_id, cat
            FROM {self.TABLE_TASK}
            WHERE cat = 'Kernel'
            ORDER BY dur DESC
        """
        if top_n:
            sql += f" LIMIT {top_n}"
        
        return self.query_sql(sql)
    
    def query_memory_usage(self) -> pd.DataFrame:
        """查询内存使用数据"""
        try:
            return self.query_table(self.TABLE_NPU_MEMORY)
        except Exception:
            logger.warning(f"表 {self.TABLE_NPU_MEMORY} 不存在")
            return pd.DataFrame()
    
    def get_overview(self) -> Dict[str, Any]:
        """
        获取数据库概览信息
        
        Returns:
            {
                "tables": [表名列表],
                "step_count": Step 数量,
                "total_time_ms": 总时间,
                ...
            }
        """
        overview = {
            "db_path": str(self.db_path),
            "tables": self.get_tables(),
        }
        
        # 尝试获取 Step 信息
        try:
            step_time = self.query_step_time()
            if not step_time.empty:
                overview["step_count"] = len(step_time)
                total_ns = (step_time["endNs"] - step_time["startNs"]).sum()
                overview["total_time_ms"] = total_ns / 1e6
        except Exception as e:
            logger.debug(f"获取 Step 信息失败: {e}")
        
        return overview


class ClusterDBQuery:
    """
    集群分析数据库查询（cluster_analysis.db）
    """
    
    def __init__(self, db_path: str):
        self.db = DBQuery(db_path)
    
    def query_cluster_step_trace(self) -> pd.DataFrame:
        """查询集群 Step Trace 时间数据"""
        return self.db.query_sql("""
            SELECT step, type, index_col, 
                   computing, communication, 
                   communication_not_overlapped, overlapped,
                   free, stage, bubble
            FROM ClusterStepTraceTime
        """)
    
    def query_slow_rank_stats(self) -> pd.DataFrame:
        """查询慢卡统计数据"""
        try:
            return self.db.query_table("SlowRankStats")
        except Exception:
            logger.warning("SlowRankStats 表不存在")
            return pd.DataFrame()
    
    def close(self):
        self.db.close()
