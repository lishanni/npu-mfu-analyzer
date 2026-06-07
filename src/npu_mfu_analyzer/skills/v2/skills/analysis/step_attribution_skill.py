"""Step 阶段归因技能。"""

import logging
from collections import Counter
from typing import Any, Dict, List

from npu_mfu_analyzer.skills.v2.base import BaseSkill, SkillType, SkillCategory, SkillMetadata, SkillPriority, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class StepAttributionSkill(BaseSkill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="attribute_step_bottleneck",
            display_name="Step 阶段归因",
            description="识别当前 profiling 最痛的 step 阶段",
            skill_type=SkillType.DIAGNOSE,
            category=SkillCategory.DIAGNOSIS,
            priority=SkillPriority.HIGH,
            tags=["step", "phase", "diagnosis"],
            dependencies=["attribute_memory_bottleneck", "diagnose_communication_exposure"],
        )

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            summary = self._to_dict(context.profiling_summary)
            hints = summary.get("training_hints", {}) or {}
            memory_result = context.get_previous_result("attribute_memory_bottleneck")
            comm_result = context.get_previous_result("diagnose_communication_exposure")

            avg_step = self._to_float(summary.get("avg_step_time"))
            avg_compute = self._to_float(summary.get("avg_compute_time"))
            avg_free = self._to_float(summary.get("avg_free_time"))
            bubble_ratio = self._to_float(summary.get("bubble_ratio"))
            free_ratio = (avg_free / avg_step) if avg_step > 0 else 0.0
            compute_ratio = (avg_compute / avg_step) if avg_step > 0 else 0.0

            memory_code = memory_result.data.get("diagnosis_code", "") if memory_result and memory_result.success else ""
            comm_code = comm_result.data.get("diagnosis_code", "") if comm_result and comm_result.success else ""
            comm_ratio = self._extract_comm_ratio(summary, context)
            host_gap = self._analyze_host_gap(context=context, free_ratio=free_ratio, comm_ratio=comm_ratio, comm_code=comm_code)

            phase = "unknown"
            confidence = 0.35
            evidence: List[str] = []
            candidate_actions: List[Dict[str, Any]] = []

            if bubble_ratio >= 20 or comm_code == "PIPELINE_BUBBLE_BOUND":
                phase = "pipeline_schedule"
                confidence = 0.85
                evidence.append(f"bubble 占比 {bubble_ratio:.1f}%")
            elif memory_code == "STATE_MEMORY_BOUND":
                phase = "optimizer_step"
                confidence = 0.75
                evidence.append("state-bound 往往在 optimizer.step 或状态常驻阶段最痛")
            elif comm_code == "HSDP_DP_COMM_EXPOSED":
                phase = "grad_state"
                confidence = 0.75
                evidence.append("DP/HSDP 通信通常暴露在 backward 尾部或 grad-state 阶段")
            elif memory_code == "ACTIVATION_MEMORY_BOUND":
                phase = "backward" if hints.get("long_context_likely") or hints.get("activation_heavy_likely") else "forward"
                confidence = 0.70
                evidence.append("activation-bound 通常首先压在 forward/backward 主计算链")
            elif comm_code in {"TP_COMM_EXPOSED", "CP_COMM_EXPOSED", "EP_COMM_EXPOSED"}:
                phase = "forward_backward"
                confidence = 0.65
                evidence.append(f"{comm_code} 更像主计算链上的通信暴露")
            elif host_gap["should_classify"]:
                phase = "input_host_gap"
                confidence = host_gap["confidence"]
                evidence.extend(host_gap["evidence"])
                candidate_actions = host_gap["actions"]
            elif compute_ratio >= 0.60:
                phase = "forward_backward"
                confidence = 0.50
                evidence.append(f"计算占比 {compute_ratio * 100:.1f}%")

            return SkillResult(
                skill_name=self.name,
                skill_type=self.skill_type,
                success=True,
                data={
                    "dominant_step_phase": phase,
                    "observation_facts": evidence,
                    "candidate_actions": candidate_actions,
                    "host_gap_evidence": host_gap["host_evidence"],
                },
                summary=f"Step 归因: {phase}",
                details=evidence,
                evidence=evidence,
                confidence=confidence,
                priority="P1" if phase != "unknown" else "P2",
            )
        except Exception as e:
            logger.exception("Step 阶段归因失败: %s", e)
            return SkillResult(skill_name=self.name, skill_type=self.skill_type, success=False, error=str(e))

    def _to_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value if isinstance(value, dict) else {}

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _extract_comm_ratio(self, summary: Dict[str, Any], context: SkillContext) -> float:
        comm_ratio = self._to_float(summary.get("comm_ratio"))
        if comm_ratio > 1.0:
            comm_ratio /= 100.0
        if comm_ratio > 0:
            return comm_ratio

        agent_results = context.user_inputs.get("agent_results", {})
        communication = agent_results.get("communication")
        details = getattr(communication, "details", {}) if communication else {}
        comm_ratio = self._to_float(details.get("comm_ratio"))
        if comm_ratio > 1.0:
            comm_ratio /= 100.0
        return comm_ratio

    def _analyze_host_gap(self, context: SkillContext, free_ratio: float, comm_ratio: float, comm_code: str) -> Dict[str, Any]:
        source_analysis = context.user_inputs.get("source_analysis", {}) or {}
        root_cause_findings = context.user_inputs.get("root_cause_findings", []) or []
        host_device_chains = context.host_device_chains or context.user_inputs.get("host_device_chains", []) or []

        source_counts = Counter()
        for chain in host_device_chains:
            source_type = ""
            if hasattr(chain, "source_type"):
                source_type = getattr(chain, "source_type", "") or ""
            elif isinstance(chain, dict):
                source_type = chain.get("source_type", "") or ""
            if source_type:
                source_counts[source_type] += 1

        total_chains = sum(source_counts.values())
        dataloader_ratio = (source_counts.get("dataloader", 0) / total_chains) if total_chains > 0 else 0.0

        host_issue_texts: List[str] = []
        for issue in source_analysis.get("potential_issues", [])[:3]:
            if self._is_host_hint(issue):
                host_issue_texts.append(str(issue))

        root_cause_host_hits: List[str] = []
        for finding in root_cause_findings[:5]:
            text_blocks = []
            if isinstance(finding, dict):
                text_blocks.extend([finding.get("rule_name", ""), finding.get("root_cause", "")])
                text_blocks.extend(finding.get("evidence", []) or [])
                text_blocks.extend(finding.get("optimization_suggestions", []) or [])
            else:
                text_blocks.extend([getattr(finding, "rule_name", ""), getattr(finding, "root_cause", "")])
                text_blocks.extend(getattr(finding, "evidence", []) or [])
                text_blocks.extend(getattr(finding, "optimization_suggestions", []) or [])
            merged = " ".join(str(item) for item in text_blocks if item)
            if merged and self._is_host_hint(merged):
                root_cause_host_hits.append(merged)

        host_evidence = {
            "free_ratio": free_ratio,
            "comm_ratio": comm_ratio,
            "source_counts": dict(source_counts),
            "dataloader_ratio": dataloader_ratio,
            "potential_issues": host_issue_texts,
            "root_cause_hits": root_cause_host_hits[:2],
        }

        supporting_score = 0.0
        evidence: List[str] = []
        if free_ratio >= 0.20:
            supporting_score += 0.35
            evidence.append(f"空闲占比 {free_ratio * 100:.1f}%")
        if comm_ratio <= 0.20 and comm_code not in {
            "TP_COMM_EXPOSED",
            "CP_COMM_EXPOSED",
            "EP_COMM_EXPOSED",
            "HSDP_DP_COMM_EXPOSED",
            "PIPELINE_BUBBLE_BOUND",
        }:
            supporting_score += 0.20
            evidence.append(f"通信占比 {comm_ratio * 100:.1f}% 不足以支撑 comm-bound 结论")
        if dataloader_ratio >= 0.10:
            supporting_score += 0.25
            evidence.append(f"Host-Device 调用链中数据加载/预取来源占比 {dataloader_ratio * 100:.1f}%")
        if host_issue_texts:
            supporting_score += 0.10
            evidence.append(f"Host 侧分析提示: {host_issue_texts[0]}")
        if root_cause_host_hits:
            supporting_score += 0.10
            evidence.append(f"根因分析提示: {root_cause_host_hits[0]}")

        should_classify = supporting_score >= 0.55 or (free_ratio >= 0.60 and comm_ratio <= 0.20)
        confidence = min(max(supporting_score, 0.45 if should_classify else 0.0), 0.85)

        actions = []
        if should_classify:
            actions = [
                {
                    "action_code": "inspect_dataloader_and_prefetch",
                    "title": "优先检查数据加载与 Host 侧预取链路",
                    "description": "结合 Host-Device 调用链、DataLoader/prefetch 栈和空闲区间，定位数据准备是否晚于计算消费点。",
                    "expected_effect": "缩短 step 中的 Host 输入间隙，降低 free/idle 时间。",
                    "rationale": "当前高空闲占比更像输入供给或 Host 侧调度间隙，而不是通信主导。",
                    "priority": "high",
                    "anti_actions": ["不先改 Comm Fusion", "不先提高 TP/PP 并行度"],
                },
                {
                    "action_code": "audit_host_device_sync",
                    "title": "排查 Host-Device 同步点和阻塞拷贝",
                    "description": "检查是否存在过早 wait、同步拷贝、Host 侧串行准备或未命中 overlap 的输入搬运。",
                    "expected_effect": "减少 Host 侧阻塞等待，提升 compute/comm overlap 的有效性。",
                    "rationale": "空闲时间高且通信占比低，优先看 Host 端同步/输入链路是否卡主关键路径。",
                    "priority": "high",
                },
                {
                    "action_code": "enable_input_pin_memory_prefetch",
                    "title": "评估 pin_memory / prefetch / worker 配置",
                    "description": "结合 DataLoader worker、pin_memory、prefetch factor 和输入 staging，做最小 A/B 试验。",
                    "expected_effect": "若是输入供给不足，可直接缩短 input-host gap。",
                    "rationale": "现有 host-bound 证据表明，优先验证输入链路配置比调整并行策略更直接。",
                    "priority": "medium",
                },
            ]

        return {
            "should_classify": should_classify,
            "confidence": confidence,
            "evidence": evidence,
            "actions": actions,
            "host_evidence": host_evidence,
        }

    def _is_host_hint(self, text: str) -> bool:
        lowered = str(text).lower()
        keywords = [
            "host-device",
            "host 侧",
            "host端",
            "host 端",
            "dataloader",
            "dataset",
            "prefetch",
            "pin_memory",
            "synchronize",
            "sync",
            "数据加载",
            "输入",
            "拷贝",
            "copy",
            "iterator",
        ]
        return any(keyword in lowered for keyword in keywords)
