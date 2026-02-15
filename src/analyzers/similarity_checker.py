"""
Profiling 相似度检测器

判断两个 Profiling 数据是否适合进行对比分析。
通过对比硬件配置、模型结构、训练框架、数据形状等维度，
给出相似度评分和对比可行性判断。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.data_loader.profiling_loader import ProfilingLoader, ProfilingInfo
from src.data_loader.data_summarizer import DataSummarizer, ProfilingSummary

logger = logging.getLogger(__name__)


class ComparabilityLevel(Enum):
    """对比可行性等级"""
    COMPARABLE = "comparable"           # 适合对比（score >= 0.6）
    PARTIALLY_COMPARABLE = "partial"    # 部分可比（0.3 <= score < 0.6）
    NOT_COMPARABLE = "not_comparable"   # 不适合对比（score < 0.3）


@dataclass
class SimilarityDimension:
    """单维度相似度评分"""
    name: str
    score: float           # 0.0 ~ 1.0
    weight: float          # 权重
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class SimilarityResult:
    """相似度检测结果"""
    overall_score: float                       # 0.0 ~ 1.0
    level: ComparabilityLevel
    dimensions: List[SimilarityDimension] = field(default_factory=list)
    summary: str = ""
    warnings: List[str] = field(default_factory=list)

    def is_comparable(self) -> bool:
        return self.level in (ComparabilityLevel.COMPARABLE, ComparabilityLevel.PARTIALLY_COMPARABLE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "level": self.level.value,
            "summary": self.summary,
            "warnings": self.warnings,
            "dimensions": [
                {
                    "name": d.name,
                    "score": round(d.score, 3),
                    "weight": d.weight,
                    "evidence": d.evidence,
                    "warnings": d.warnings,
                }
                for d in self.dimensions
            ],
        }

    def to_markdown(self) -> str:
        """转为 Markdown 描述"""
        level_map = {
            ComparabilityLevel.COMPARABLE: "适合对比",
            ComparabilityLevel.PARTIALLY_COMPARABLE: "部分可比（请注意差异）",
            ComparabilityLevel.NOT_COMPARABLE: "不建议对比",
        }
        icon_map = {
            ComparabilityLevel.COMPARABLE: "✅",
            ComparabilityLevel.PARTIALLY_COMPARABLE: "⚠️",
            ComparabilityLevel.NOT_COMPARABLE: "❌",
        }
        lines = [
            f"### 相似度评估 {icon_map[self.level]}",
            "",
            f"**总体评分**: {self.overall_score * 100:.0f}%  |  **结论**: {level_map[self.level]}",
            "",
        ]

        if self.summary:
            lines.append(f"> {self.summary}")
            lines.append("")

        lines.append("| 维度 | 得分 | 权重 | 说明 |")
        lines.append("|------|------|------|------|")
        for d in self.dimensions:
            evidence_text = "; ".join(d.evidence[:2]) if d.evidence else "-"
            lines.append(
                f"| {d.name} | {d.score * 100:.0f}% | {d.weight * 100:.0f}% | {evidence_text} |"
            )

        if self.warnings:
            lines.append("")
            lines.append("**注意事项:**")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")

        return "\n".join(lines)


class SimilarityChecker:
    """
    Profiling 相似度检测器

    通过多个维度评估两个 Profiling 数据的相似程度，
    决定是否适合进行对比分析。

    维度及权重:
    - 硬件匹配 (0.35): 芯片类型、AICore 数量、卡数
    - 模型匹配 (0.30): 算子名称重合度、模型架构、层数
    - 框架匹配 (0.15): 训练框架类型
    - 数据形状 (0.20): Step 数量范围、Step 时间量级
    """

    # 相似度维度权重
    WEIGHTS = {
        "hardware": 0.35,
        "model": 0.30,
        "framework": 0.15,
        "data_shape": 0.20,
    }

    # 判定阈值
    THRESHOLD_COMPARABLE = 0.6
    THRESHOLD_PARTIAL = 0.3

    def __init__(self):
        pass

    def check(
        self,
        info_a: ProfilingInfo,
        info_b: ProfilingInfo,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
        operators_a: Optional[List[Dict[str, Any]]] = None,
        operators_b: Optional[List[Dict[str, Any]]] = None,
    ) -> SimilarityResult:
        """
        执行相似度检测

        Args:
            info_a: Profiling A 的基本信息
            info_b: Profiling B 的基本信息
            summary_a: Profiling A 的摘要
            summary_b: Profiling B 的摘要
            operators_a: Profiling A 的算子列表（可选）
            operators_b: Profiling B 的算子列表（可选）

        Returns:
            SimilarityResult
        """
        dimensions = []

        # 1. 硬件匹配
        hw_dim = self._check_hardware(info_a, info_b)
        dimensions.append(hw_dim)

        # 2. 模型匹配
        model_dim = self._check_model(
            summary_a, summary_b,
            operators_a or [], operators_b or []
        )
        dimensions.append(model_dim)

        # 3. 框架匹配
        fw_dim = self._check_framework(info_a, info_b)
        dimensions.append(fw_dim)

        # 4. 数据形状匹配
        ds_dim = self._check_data_shape(summary_a, summary_b)
        dimensions.append(ds_dim)

        # 计算总分
        overall_score = sum(d.weighted_score for d in dimensions)
        overall_score = max(0.0, min(1.0, overall_score))

        # 判定等级
        if overall_score >= self.THRESHOLD_COMPARABLE:
            level = ComparabilityLevel.COMPARABLE
        elif overall_score >= self.THRESHOLD_PARTIAL:
            level = ComparabilityLevel.PARTIALLY_COMPARABLE
        else:
            level = ComparabilityLevel.NOT_COMPARABLE

        # 收集所有警告
        all_warnings = []
        for d in dimensions:
            all_warnings.extend(d.warnings)

        # 生成总结
        summary = self._generate_summary(level, dimensions, all_warnings)

        return SimilarityResult(
            overall_score=overall_score,
            level=level,
            dimensions=dimensions,
            summary=summary,
            warnings=all_warnings,
        )

    def _check_hardware(
        self,
        info_a: ProfilingInfo,
        info_b: ProfilingInfo,
    ) -> SimilarityDimension:
        """检查硬件相似度"""
        score = 0.0
        evidence = []
        warnings = []

        # 检查数据类型（DB vs JSON）
        if info_a.data_type == info_b.data_type:
            score += 0.1
            evidence.append(f"数据格式相同: {info_a.data_type}")

        # 检查 Rank 数量
        if info_a.rank_count == info_b.rank_count:
            score += 0.5
            evidence.append(f"卡数相同: {info_a.rank_count}")
        elif info_a.rank_count > 0 and info_b.rank_count > 0:
            ratio = min(info_a.rank_count, info_b.rank_count) / max(info_a.rank_count, info_b.rank_count)
            if ratio >= 0.5:
                score += 0.3
                evidence.append(f"卡数接近: {info_a.rank_count} vs {info_b.rank_count}")
                warnings.append(f"卡数不同 ({info_a.rank_count} vs {info_b.rank_count})，并行策略可能不同")
            else:
                score += 0.1
                warnings.append(f"卡数差异较大 ({info_a.rank_count} vs {info_b.rank_count})")

        # 检查硬件信息是否可用（通过 info.json）
        # 这里我们通过路径中的设备类型来推断
        hw_a = self._extract_hardware_hint(info_a)
        hw_b = self._extract_hardware_hint(info_b)

        if hw_a and hw_b:
            if hw_a == hw_b:
                score += 0.4
                evidence.append(f"硬件类型相同: {hw_a}")
            else:
                score += 0.1
                warnings.append(f"硬件类型不同: {hw_a} vs {hw_b}")
        else:
            # 无法确认硬件类型，给予中等分数
            score += 0.2
            evidence.append("硬件类型无法确认，假设相同")

        score = min(score, 1.0)
        return SimilarityDimension(
            name="硬件匹配",
            score=score,
            weight=self.WEIGHTS["hardware"],
            evidence=evidence,
            warnings=warnings,
        )

    def _check_model(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
        operators_a: List[Dict[str, Any]],
        operators_b: List[Dict[str, Any]],
    ) -> SimilarityDimension:
        """检查模型相似度"""
        score = 0.0
        evidence = []
        warnings = []

        # 通过算子名称集合计算 Jaccard 相似度
        names_a = {op.get("name", "") for op in operators_a if op.get("name")}
        names_b = {op.get("name", "") for op in operators_b if op.get("name")}

        if names_a and names_b:
            intersection = names_a & names_b
            union = names_a | names_b
            jaccard = len(intersection) / len(union) if union else 0

            if jaccard >= 0.7:
                score += 0.7
                evidence.append(f"算子重合度高: {jaccard * 100:.0f}% ({len(intersection)}/{len(union)})")
            elif jaccard >= 0.4:
                score += 0.4
                evidence.append(f"算子重合度中: {jaccard * 100:.0f}% ({len(intersection)}/{len(union)})")
                warnings.append("算子名称有一定差异，模型结构可能已改变")
            elif jaccard >= 0.2:
                score += 0.2
                evidence.append(f"算子重合度低: {jaccard * 100:.0f}%")
                warnings.append("算子名称差异较大，可能是不同模型")
            else:
                evidence.append(f"算子几乎无重合: {jaccard * 100:.0f}%")
                warnings.append("算子完全不同，很可能是不同模型")

            # 检查是否有新增/消失的关键算子类型
            new_ops = names_b - names_a
            removed_ops = names_a - names_b
            if new_ops:
                evidence.append(f"新增算子: {len(new_ops)} 个")
            if removed_ops:
                evidence.append(f"移除算子: {len(removed_ops)} 个")
        else:
            # 无算子数据，使用 top_operators 进行简化比较
            top_a = {op.get("name", "") for op in summary_a.top_operators if op.get("name")}
            top_b = {op.get("name", "") for op in summary_b.top_operators if op.get("name")}

            if top_a and top_b:
                intersection = top_a & top_b
                union = top_a | top_b
                jaccard = len(intersection) / len(union) if union else 0

                if jaccard >= 0.5:
                    score += 0.5
                    evidence.append(f"Top 算子重合度: {jaccard * 100:.0f}%")
                elif jaccard >= 0.2:
                    score += 0.3
                    evidence.append(f"Top 算子有部分重合: {jaccard * 100:.0f}%")
                else:
                    score += 0.1
                    warnings.append("Top 算子差异很大")
            else:
                score += 0.3  # 无数据，给中等分
                evidence.append("无算子数据可比较")

        # 额外：检查时间量级是否在同一数量级（间接判断模型规模）
        if summary_a.avg_step_time > 0 and summary_b.avg_step_time > 0:
            time_ratio = min(summary_a.avg_step_time, summary_b.avg_step_time) / \
                         max(summary_a.avg_step_time, summary_b.avg_step_time)
            if time_ratio >= 0.3:
                score += 0.3
                evidence.append(f"Step 时间量级接近 (ratio={time_ratio:.2f})")
            else:
                score += 0.1
                warnings.append(f"Step 时间量级差异大 ({summary_a.avg_step_time / 1000:.1f}ms vs {summary_b.avg_step_time / 1000:.1f}ms)")

        score = min(score, 1.0)
        return SimilarityDimension(
            name="模型匹配",
            score=score,
            weight=self.WEIGHTS["model"],
            evidence=evidence,
            warnings=warnings,
        )

    def _check_framework(
        self,
        info_a: ProfilingInfo,
        info_b: ProfilingInfo,
    ) -> SimilarityDimension:
        """检查框架相似度"""
        score = 0.0
        evidence = []
        warnings = []

        if info_a.framework and info_b.framework:
            if info_a.framework == info_b.framework:
                score = 1.0
                evidence.append(f"框架相同: {info_a.framework}")
            else:
                score = 0.3
                evidence.append(f"框架不同: {info_a.framework} vs {info_b.framework}")
                warnings.append(f"训练框架不同 ({info_a.framework} vs {info_b.framework})，对比结果需谨慎解读")
        else:
            score = 0.5
            evidence.append("框架信息不完整")

        return SimilarityDimension(
            name="框架匹配",
            score=score,
            weight=self.WEIGHTS["framework"],
            evidence=evidence,
            warnings=warnings,
        )

    def _check_data_shape(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
    ) -> SimilarityDimension:
        """检查数据形状相似度"""
        score = 0.0
        evidence = []
        warnings = []

        # Step 数量比较
        if summary_a.step_count > 0 and summary_b.step_count > 0:
            step_ratio = min(summary_a.step_count, summary_b.step_count) / \
                         max(summary_a.step_count, summary_b.step_count)
            if step_ratio >= 0.5:
                score += 0.4
                evidence.append(f"Step 数量接近: {summary_a.step_count} vs {summary_b.step_count}")
            elif step_ratio >= 0.2:
                score += 0.2
                evidence.append(f"Step 数量有差异: {summary_a.step_count} vs {summary_b.step_count}")
            else:
                score += 0.1
                warnings.append(f"Step 数量差异极大: {summary_a.step_count} vs {summary_b.step_count}")
        else:
            score += 0.2
            evidence.append("Step 数量信息不完整")

        # Rank 数量一致性（补充硬件检查）
        if summary_a.rank_count == summary_b.rank_count:
            score += 0.3
        elif summary_a.rank_count > 0 and summary_b.rank_count > 0:
            score += 0.1

        # 时间分布模式相似度
        # 检查 compute/comm/free 比例是否在相似范围内
        ratios_similar = self._check_ratio_similarity(summary_a, summary_b)
        if ratios_similar:
            score += 0.3
            evidence.append("时间分布模式相似")
        else:
            score += 0.1
            evidence.append("时间分布模式有差异")

        score = min(score, 1.0)
        return SimilarityDimension(
            name="数据形状",
            score=score,
            weight=self.WEIGHTS["data_shape"],
            evidence=evidence,
            warnings=warnings,
        )

    def _check_ratio_similarity(
        self,
        summary_a: ProfilingSummary,
        summary_b: ProfilingSummary,
    ) -> bool:
        """检查时间分布比例是否相似"""
        total_a = summary_a.avg_compute_time + summary_a.avg_comm_time + summary_a.avg_free_time
        total_b = summary_b.avg_compute_time + summary_b.avg_comm_time + summary_b.avg_free_time

        if total_a <= 0 or total_b <= 0:
            return True  # 无数据时假设相似

        # 计算各项占比
        ratios_a = [
            summary_a.avg_compute_time / total_a,
            summary_a.avg_comm_time / total_a,
            summary_a.avg_free_time / total_a,
        ]
        ratios_b = [
            summary_b.avg_compute_time / total_b,
            summary_b.avg_comm_time / total_b,
            summary_b.avg_free_time / total_b,
        ]

        # 各项差异不超过 30 个百分点视为相似
        for ra, rb in zip(ratios_a, ratios_b):
            if abs(ra - rb) > 0.30:
                return False
        return True

    def _extract_hardware_hint(self, info: ProfilingInfo) -> Optional[str]:
        """从 Profiling 路径中提取硬件类型提示"""
        import re
        path_lower = info.path.lower()

        # 尝试从路径匹配常见硬件型号
        patterns = [
            (r"atlas.?a2", "Atlas A2"),
            (r"atlas.?300i", "Atlas 300I"),
            (r"910b", "Ascend 910B"),
            (r"910a", "Ascend 910A"),
            (r"910pro", "Ascend 910 Pro"),
            (r"310p", "Ascend 310P"),
        ]

        for pattern, name in patterns:
            if re.search(pattern, path_lower):
                return name

        # 尝试从 info.json 读取（如果存在）
        try:
            from pathlib import Path
            import json
            info_files = list(Path(info.path).glob("**/info.json"))
            if not info_files:
                info_files = list(Path(info.path).glob("**/device_*/info.json"))
            if info_files:
                with open(str(info_files[0]), "r") as f:
                    device_info = json.load(f)
                    soc_name = device_info.get("SocInfo", {}).get("SocName", "")
                    if soc_name:
                        return soc_name
        except Exception:
            pass

        return None

    def _generate_summary(
        self,
        level: ComparabilityLevel,
        dimensions: List[SimilarityDimension],
        warnings: List[str],
    ) -> str:
        """生成总结文本"""
        if level == ComparabilityLevel.COMPARABLE:
            summary = "两个 Profiling 数据高度相似，适合进行对比分析。"
        elif level == ComparabilityLevel.PARTIALLY_COMPARABLE:
            # 找出得分最低的维度
            lowest = min(dimensions, key=lambda d: d.score)
            summary = (
                f"两个 Profiling 数据部分可比，但在「{lowest.name}」维度存在差异。"
                f"对比结果仅供参考。"
            )
        else:
            # 找出得分最低的维度
            low_dims = [d for d in dimensions if d.score < 0.3]
            dim_names = "、".join(d.name for d in low_dims) if low_dims else "多个维度"
            summary = (
                f"两个 Profiling 数据在{dim_names}上差异较大，"
                f"不建议进行对比分析。请确认两次 Profiling 是否来自同一训练任务。"
            )
        return summary
