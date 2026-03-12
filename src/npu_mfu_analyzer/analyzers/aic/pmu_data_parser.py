"""
AIC PMU 数据解析器

从 msprof 采集的 profiling 数据中解析 AIC (AI Core) PMU 事件。
匹配 msprof 框架的 PMU 数据结构。
"""

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd

from npu_mfu_analyzer.data_loader.aic_metrics import (
    ExtendedAICMetrics,
    ExtendedArithmeticUtilization,
    ExtendedMemoryMetrics,
    ExtendedPipelineMetrics,
    AICMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class PMUEvent:
    """PMU 事件定义"""
    name: str
    description: str
    unit: str
    scale_factor: float = 1.0


# msprof AIC PMU 事件定义（匹配 msprof 框架）
AIC_PMU_EVENTS = {
    # 算术单元事件
    "AI_CORE_ARITHMETIC_UTILIZATION": PMUEvent(
        name="cube_utilization",
        description="Cube 单元利用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_VECTOR_UTILIZATION": PMUEvent(
        name="vector_utilization",
        description="Vector 单元利用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_SCALAR_UTILIZATION": PMUEvent(
        name="scalar_utilization",
        description="Scalar 单元利用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_INSTRUCTION_ISSUE_RATE": PMUEvent(
        name="instruction_issue_rate",
        description="指令发射率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_CUBE_ACTIVE_CYCLES": PMUEvent(
        name="cube_active_cycles",
        description="Cube 活跃周期数",
        unit="cycles",
        scale_factor=1.0,
    ),
    "AI_CORE_VECTOR_ACTIVE_CYCLES": PMUEvent(
        name="vector_active_cycles",
        description="Vector 活跃周期数",
        unit="cycles",
        scale_factor=1.0,
    ),

    # 内存访问事件
    "AI_CORE_L2_CACHE_HIT_RATE": PMUEvent(
        name="l2_cache_hit_rate",
        description="L2 缓存命中率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_L2_READ_BANDWIDTH": PMUEvent(
        name="l2_read_bandwidth",
        description="L2 读带宽",
        unit="GB/s",
        scale_factor=1.0,
    ),
    "AI_CORE_L2_WRITE_BANDWIDTH": PMUEvent(
        name="l2_write_bandwidth",
        description="L2 写带宽",
        unit="GB/s",
        scale_factor=1.0,
    ),
    "AI_CORE_UB_USAGE": PMUEvent(
        name="ub_usage",
        description="Unified Buffer 使用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_L0A_UTILIZATION": PMUEvent(
        name="l0a_utilization",
        description="L0A Buffer 使用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_L0B_UTILIZATION": PMUEvent(
        name="l0b_utilization",
        description="L0B Buffer 使用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_L0C_UTILIZATION": PMUEvent(
        name="l0c_utilization",
        description="L0C Buffer 使用率",
        unit="%",
        scale_factor=1.0,
    ),

    # 流水线事件
    "AI_CORE_PIPE_UTILIZATION": PMUEvent(
        name="pipe_utilization",
        description="流水线利用率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_STALL_RATE": PMUEvent(
        name="stall_rate",
        description="停顿率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_MTE_STALL_RATE": PMUEvent(
        name="mte_stall_rate",
        description="MTE 停顿率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_DEPENDENCY_STALL_RATE": PMUEvent(
        name="dependency_stall_rate",
        description="依赖停顿率",
        unit="%",
        scale_factor=1.0,
    ),
    "AI_CORE_MEMORY_STALL_RATE": PMUEvent(
        name="memory_stall_rate",
        description="内存停顿率",
        unit="%",
        scale_factor=1.0,
    ),

    # 资源冲突事件
    "AI_CORE_RESOURCE_CONFLICT_RATIO": PMUEvent(
        name="resource_conflict_ratio",
        description="资源冲突率",
        unit="%",
        scale_factor=1.0,
    ),
}


class PMUDataParser:
    """
    PMU 数据解析器

    从 msprof 采集的 profiling 数据中解析 AIC PMU 事件。
    支持从 SQLite 数据库读取 PMU 指标。
    """

    def __init__(self, profiling_path: str):
        """
        初始化 PMU 数据解析器

        Args:
            profiling_path: Profiling 数据目录路径
        """
        self.profiling_path = Path(profiling_path)
        self._db_path: Optional[Path] = None
        self._pmu_events_map = {v.name: v for k, v in AIC_PMU_EVENTS.items()}

    def find_db(self) -> Optional[Path]:
        """
        查找 profiling 数据库文件

        Returns:
            数据库文件路径，未找到返回 None
        """
        # 查找 MetricsSummary 数据库
        db_files = list(self.profiling_path.rglob("*MetricSummary*.db"))
        if db_files:
            self._db_path = db_files[0]
            return self._db_path

        # 查找其他可能的数据库
        db_files = list(self.profiling_path.rglob("*.db"))
        if db_files:
            self._db_path = db_files[0]
            return self._db_path

        logger.warning(f"No database file found in {self.profiling_path}")
        return None

    def load_pmu_data(
        self,
        step_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        加载 PMU 数据

        Args:
            step_ids: 要加载的 step ID 列表，None 表示全部

        Returns:
            PMU 数据 DataFrame
        """
        if not self._db_path:
            self.find_db()

        if not self._db_path:
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(str(self._db_path))

            # 检查是否有 metrics_summary 表
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='metrics_summary'"
            )

            if not cursor.fetchone():
                logger.warning(f"metrics_summary table not found in {self._db_path}")
                return pd.DataFrame()

            # 构建查询
            query = "SELECT * FROM metrics_summary"

            if step_ids:
                placeholders = ",".join("?" * len(step_ids))
                # 检查是否有 step_id 列
                cursor.execute("PRAGMA table_info(metrics_summary)")
                columns = [row[1] for row in cursor.fetchall()]
                if "step_id" in columns:
                    query += f" WHERE step_id IN ({placeholders})"
                    df = pd.read_sql_query(query, conn, params=step_ids)
                else:
                    df = pd.read_sql_query(query, conn)
            else:
                df = pd.read_sql_query(query, conn)

            conn.close()

            logger.debug(f"Loaded {len(df)} PMU records from {self._db_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to load PMU data: {e}")
            return pd.DataFrame()

    def parse_operator_pmu(
        self,
        operator_name: str,
        pmu_row: pd.Series,
        duration_us: Optional[float] = None,
    ) -> ExtendedAICMetrics:
        """
        解析单个算子的 PMU 数据

        Args:
            operator_name: 算子名称
            pmu_row: PMU 数据行
            duration_us: 算子执行时间（微秒）

        Returns:
            ExtendedAICMetrics: 扩展 AIC 指标
        """
        metrics = AICMetrics(
            op_name=operator_name,
            op_type="unknown",
            duration_us=duration_us or 0.0,
        )

        # 解析算术单元指标
        if self._has_pmu_data(pmu_row, "cube_utilization"):
            metrics.arithmetic = ArithmeticUtilization(
                cube_utilization=pmu_row.get("cube_utilization", 0.0),
                vector_utilization=pmu_row.get("vector_utilization", 0.0),
                scalar_utilization=pmu_row.get("scalar_utilization", 0.0),
                total_cycles=pmu_row.get("total_cycles", 0),
            )

        # 解析内存指标
        if self._has_pmu_data(pmu_row, "l2_cache_hit_rate"):
            metrics.memory = MemoryMetrics(
                l2_cache_hit_rate=pmu_row.get("l2_cache_hit_rate", 0.0),
                l2_read_bandwidth=pmu_row.get("l2_read_bandwidth", 0.0),
                l2_write_bandwidth=pmu_row.get("l2_write_bandwidth", 0.0),
                ub_usage=pmu_row.get("ub_usage", 0.0),
                l0_usage=pmu_row.get("l0a_utilization", 0.0),
            )

        # 解析流水线指标
        if self._has_pmu_data(pmu_row, "pipe_utilization"):
            metrics.pipeline = PipelineMetrics(
                pipe_utilization=pmu_row.get("pipe_utilization", 0.0),
                stall_rate=pmu_row.get("stall_rate", 0.0),
                resource_conflict_ratio=pmu_row.get("resource_conflict_ratio", 0.0),
            )

        # 转换为扩展指标
        extended = ExtendedAICMetrics(
            op_name=metrics.op_name,
            op_type=metrics.op_type,
            duration_us=metrics.duration_us,
            raw_data=pmu_row.to_dict(),
        )

        if metrics.arithmetic:
            extended.arithmetic = metrics.arithmetic

        if metrics.memory:
            extended.memory = metrics.memory

        if metrics.pipeline:
            extended.pipeline = metrics.pipeline

        # 解析扩展指标
        extended.extended_arithmetic = self._parse_extended_arithmetic(pmu_row)
        extended.extended_memory = self._parse_extended_memory(pmu_row)
        extended.extended_pipeline = self._parse_extended_pipeline(pmu_row)

        return extended

    def _has_pmu_data(self, row: pd.Series, metric_name: str) -> bool:
        """检查是否存在 PMU 数据"""
        return metric_name in row and pd.notna(row[metric_name])

    def _parse_extended_arithmetic(
        self,
        row: pd.Series,
    ) -> Optional[ExtendedArithmeticUtilization]:
        """解析扩展算术单元指标"""
        if not self._has_pmu_data(row, "cube_utilization"):
            return None

        return ExtendedArithmeticUtilization(
            cube_utilization=row.get("cube_utilization", 0.0),
            vector_utilization=row.get("vector_utilization", 0.0),
            scalar_utilization=row.get("scalar_utilization", 0.0),
            total_cycles=row.get("total_cycles", 0),
            cube_instructions=row.get("cube_instructions", 0),
            vector_instructions=row.get("vector_instructions", 0),
            scalar_instructions=row.get("scalar_instructions", 0),
            instruction_issue_rate=row.get("instruction_issue_rate", 0.0),
            cube_active_cycles=row.get("cube_active_cycles", 0),
            vector_active_cycles=row.get("vector_active_cycles", 0),
            cube_efficiency=row.get("cube_efficiency", 0.0),
            vector_efficiency=row.get("vector_efficiency", 0.0),
        )

    def _parse_extended_memory(
        self,
        row: pd.Series,
    ) -> Optional[ExtendedMemoryMetrics]:
        """解析扩展内存指标"""
        if not self._has_pmu_data(row, "l2_cache_hit_rate"):
            return None

        return ExtendedMemoryMetrics(
            l2_cache_hit_rate=row.get("l2_cache_hit_rate", 0.0),
            l2_read_bandwidth=row.get("l2_read_bandwidth", 0.0),
            l2_write_bandwidth=row.get("l2_write_bandwidth", 0.0),
            ub_usage=row.get("ub_usage", 0.0),
            l0_usage=row.get("l0a_utilization", 0.0),
            l2_read_bytes=row.get("l2_read_bytes", 0),
            l2_write_bytes=row.get("l2_write_bytes", 0),
            l2_read_requests=row.get("l2_read_requests", 0),
            l2_write_requests=row.get("l2_write_requests", 0),
            l2_miss_count=row.get("l2_miss_count", 0),
            ub_peak_usage=row.get("ub_peak_usage", 0.0),
            ub_spill_count=row.get("ub_spill_count", 0),
            ub_spill_bytes=row.get("ub_spill_bytes", 0),
            ub_conflict_rate=row.get("ub_conflict_rate", 0.0),
            l0a_utilization=row.get("l0a_utilization", 0.0),
            l0b_utilization=row.get("l0b_utilization", 0.0),
            l0c_utilization=row.get("l0c_utilization", 0.0),
            hbm_read_bytes=row.get("hbm_read_bytes", 0),
            hbm_write_bytes=row.get("hbm_write_bytes", 0),
            hbm_access_count=row.get("hbm_access_count", 0),
            hbm_bandwidth_utilization=row.get("hbm_bandwidth_utilization", 0.0),
            locality_score=0.0,  # 计算得出
            reuse_distance=0.0,  # 计算得出
        )

    def _parse_extended_pipeline(
        self,
        row: pd.Series,
    ) -> Optional[ExtendedPipelineMetrics]:
        """解析扩展流水线指标"""
        if not self._has_pmu_data(row, "pipe_utilization"):
            return None

        return ExtendedPipelineMetrics(
            pipe_utilization=row.get("pipe_utilization", 0.0),
            stall_rate=row.get("stall_rate", 0.0),
            resource_conflict_ratio=row.get("resource_conflict_ratio", 0.0),
            mte_stall_rate=row.get("mte_stall_rate", 0.0),
            vec_stall_rate=row.get("vec_stall_rate", 0.0),
            scalar_stall_rate=row.get("scalar_stall_rate", 0.0),
            dependency_stall_rate=row.get("dependency_stall_rate", 0.0),
            memory_stall_rate=row.get("memory_stall_rate", 0.0),
            sync_stall_rate=row.get("sync_stall_rate", 0.0),
            mte_conflict_rate=row.get("mte_conflict_rate", 0.0),
            vec_conflict_rate=row.get("vec_conflict_rate", 0.0),
            ub_conflict_rate=row.get("ub_conflict_rate", 0.0),
            issue_rate=row.get("issue_rate", 0.0),
            commit_rate=row.get("commit_rate", 0.0),
            branch_misprediction_rate=row.get("branch_misprediction_rate", 0.0),
        )

    def load_operator_pmu_list(
        self,
        limit: int = 100,
    ) -> List[ExtendedAICMetrics]:
        """
        加载算子 PMU 数据列表

        Args:
            limit: 最大加载数量

        Returns:
            ExtendedAICMetrics 列表
        """
        df = self.load_pmu_data()

        if df.empty:
            return []

        # 获取前 N 个算子的 PMU 数据
        metrics_list = []

        for idx, row in df.head(limit).iterrows():
            # 尝试从不同列获取算子名称
            op_name = row.get("op_name") or row.get("name") or row.get("operator") or f"op_{idx}"

            # 尝试获取执行时间
            duration = row.get("duration") or row.get("duration_ms") or row.get("total_time_ms")
            if duration:
                duration = float(duration) * 1000 if duration < 100 else float(duration)

            metrics = self.parse_operator_pmu(op_name, row, duration)
            metrics_list.append(metrics)

        logger.info(f"Loaded {len(metrics_list)} operator PMU metrics")
        return metrics_list


def parse_pmu_data(
    profiling_path: str,
    limit: int = 100,
) -> List[ExtendedAICMetrics]:
    """
    便捷函数：解析 PMU 数据

    Args:
        profiling_path: Profiling 数据路径
        limit: 最大加载数量

    Returns:
        ExtendedAICMetrics 列表
    """
    parser = PMUDataParser(profiling_path)
    return parser.load_operator_pmu_list(limit=limit)