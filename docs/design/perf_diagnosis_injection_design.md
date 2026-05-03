# npu-mfu-analyzer 并行/内存调优知识注入设计

## 1. 背景

当前 `npu-mfu-analyzer` 已经具备以下能力：

- 多 Agent 分析：`TimelineAgent`、`MemoryAgent`、`CommunicationAgent`、`AdvisorAgent`
- 多 Analyzer 计算：overlap、bubble、communication split、slow rank、MFU 等
- Root cause rules：通信瓶颈、compile 切换、小算子过多等
- Skills v2：timeline / communication 等分析技能

但现有结论生成还存在一个明显缺口：

1. **观测结果有了，但缺少系统化归因框架**
   - 目前更偏“指标超阈值 -> 给通用建议”
   - 缺少“step 阶段 -> 对象类型 -> 三层机制”的映射
2. **主矛盾没有被显式建模**
   - 缺少 `activation-bound / state-bound / communication-bound / topology-bound / bubble-bound` 这类统一分类
3. **优先调优动作不够收敛**
   - 当前建议偏宽泛
   - 缺少“并行维 / 内存策略 / 系统优化”三层优先级
4. **对大模型训练知识的吸收还不够**
   - 还没有把 PP/TP/CP/EP/HSDP、recompute/swap/offload、overlap/prefetch/fusion 之间的诊断逻辑注入分析链

本设计文档目标是把这套知识注入 analyzer，使其输出从“分析摘要”升级为：

```text
观测结果 -> 主矛盾分类 -> 优先调优动作
```

---

## 2. 设计目标

### 2.1 目标

新增一条统一的性能诊断链，显式回答三类问题：

1. **观测结果**
   - 当前慢在哪个 step 阶段
   - 峰值显存出在哪个阶段
   - 哪类通信在暴露
   - overlap / prefetch 是否真的生效

2. **主矛盾分析**
   - 当前更像：
     - depth/width/context/expert 问题
     - activation/state 问题
     - overlap/fusion/topology 问题
   - 主矛盾属于：
     - 并行维层
     - 内存策略层
     - 系统优化层

3. **优先调优动作**
   - 当前最先改什么
   - 为什么先改它
   - 明确不建议先改什么

### 2.2 非目标

本期不做：

- 直接替换现有所有 Agent
- 自动改代码或自动调参
- 建完整训练系统知识图谱
- 完整 PP runtime 模拟

本期先做 **诊断能力增强**，不是自动优化执行器。

---

## 3. 设计原则

### 3.1 显式分层

把结论明确分到三层：

1. **并行维层**
   - PP / TP / CP / EP / HSDP
2. **内存策略层**
   - recompute / swap / parameter offload / optimizer-state-offload
3. **系统优化层**
   - overlap / prefetch / fusion / async checkpoint / compile

### 3.2 先观测，后推理

新链路必须先基于结构化指标构造事实，再做规则或 LLM 推理。

```text
先测，再猜；先定位，再建议。
```

### 3.3 输出要可执行

建议不能只说“优化通信”“降低显存”。
必须明确到：

- 优先改哪一层
- 优先改哪一个动作
- 期望改善哪个观测指标

### 3.4 兼容现有架构

优先复用：

- `TimelineAgent`
- `MemoryAgent`
- `CommunicationAgent`
- `AdvisorAgent`
- `RootCauseSkillEngine`
- `skills/v2`

避免平行再造第三套诊断系统。

---

## 4. 现状缺口映射

### 4.1 TimelineAgent 的缺口

当前已有：

- 计算/通信/空闲占比
- overlap ratio
- bubble ratio
- top operators

缺口：

- 没有 step-phase 诊断模板
- 没有把结果映射到 `forward/backward/grad-state/step/save` 主矛盾
- 没有判断“wait 太早 / 没有可重叠工作 / 通信太碎 / 拓扑不对”

### 4.2 MemoryAgent 的缺口

当前已有：

- 峰值内存
- 模型/优化器/激活/梯度等分布字段
- 碎片、泄漏、OOM 风险

缺口：

- 没有“状态爆 vs activation 爆”的统一诊断
- 没有把峰值映射到 `forward/backward/step/save`
- 没有从观测结果直接导出 `recompute / swap / optimizer-state-offload / parameter-offload` 优先级

### 4.3 CommunicationAgent 的缺口

当前已有：

- TP/DP/PP 通信拆分
- 慢卡检测
- 带宽与通信占比

缺口：

- 对 TP/CP/EP/HSDP/PP 的诊断粒度不对齐
- 没有显式的“哪类通信在 step 哪一段暴露”
- 没有与 overlap/prefetch/fusion 形成动作闭环

### 4.4 AdvisorAgent 的缺口

当前已有：

- 汇总各 Agent 结果
- 给建议

缺口：

- 缺少统一的主矛盾对象
- 建议没有建立“优先改一层”的纪律
- 输出仍偏扁平，不够工程化

---

## 5. 目标能力模型

建议新增一条统一诊断模型：

### 5.1 四类观测事实（Observation Facts）

1. **Step 级事实**
   - 哪个阶段最慢：`data/input` / `forward` / `backward` / `grad_state` / `optimizer_step` / `save`
2. **显存级事实**
   - 哪个阶段峰值最高
   - 更像 activation / grad / optimizer state / parameter residency
3. **通信级事实**
   - 哪类通信在暴露：TP / CP / EP / HSDP / PP
4. **异步级事实**
   - overlap 是否有效
   - prefetch 太早还是太晚
   - wait 是否过早

### 5.2 主矛盾分类（Main Contradiction）

建议标准化为以下枚举：

- `WIDTH_BOUND`
- `DEPTH_BOUND`
- `LONG_CONTEXT_BOUND`
- `EXPERT_BOUND`
- `ACTIVATION_MEMORY_BOUND`
- `STATE_MEMORY_BOUND`
- `COMMUNICATION_EXPOSED`
- `OVERLAP_INEFFECTIVE`
- `COMMUNICATION_FRAGMENTED`
- `PIPELINE_BUBBLE_BOUND`
- `CHECKPOINT_IO_BOUND`
- `EXECUTION_FRAGMENTED`

### 5.3 优先调优动作（Prioritized Action）

动作要归属于三层之一：

- 并行维层：`adjust_tp`, `adjust_pp`, `enable_cp`, `adjust_ep`, `adjust_hsdp`
- 内存策略层：`enable_recompute`, `adjust_recompute_granularity`, `enable_swap`, `enable_param_offload`, `enable_optimizer_state_offload`
- 系统优化层：`improve_overlap`, `tune_prefetch`, `enable_comm_fusion`, `enable_async_checkpoint`, `enable_compile_or_fused_kernel`

---

## 6. 推荐落地方案

### 6.1 总体建议

推荐采用：

```text
结构化规则链 + 技能链增强 + Advisor 汇总重写
```

不建议本期直接做一个全新的“大一统超级 Agent”。

原因：

- 当前架构已经有 Agent / Analyzer / Skills 三层
- 最低成本路径是补“中间诊断层”
- 这样风险最小，也便于逐步验证

### 6.2 注入点选择

#### 方案 A：仅增强 Advisor Prompt

做法：
- 把前面学习的调优知识写进 Advisor prompt
- 让 LLM 自由发挥

优点：
- 改动最小

缺点：
- 不稳定
- 很难测试
- 结构化结果弱

结论：**不推荐单独采用**。

#### 方案 B：增强 RootCauseSkillEngine

做法：
- 在 rule engine 中加入 step/memory/communication/overlap 规则
- 直接产出主矛盾和动作

优点：
- 可测试
- 规则清晰

缺点：
- 当前 root cause engine 偏 compare/规则文本风格
- 不适合承载所有 step/memory/communication 结构化推理

结论：**适合作为规则承载层之一，但不适合单独扛全链路**。

#### 方案 C：在 `skills/v2` 新增“诊断技能链”

做法：
- 新增一组 analysis/diagnosis skills
- 统一消化 Timeline/Memory/Communication 的结构化结果
- 输出 Observation / MainContradiction / PrioritizedAction
- Advisor 只负责格式化最终报告

优点：
- 最符合当前技能化方向
- 易测
- 易扩展
- 与现有 Agent/Advisor 分层清晰

结论：**推荐作为 MVP 主方案**。

---

## 7. 具体设计

### 7.1 新增数据对象

建议新增统一诊断对象，例如：

```python
@dataclass
class ObservationFact:
    category: str              # step/memory/communication/async
    name: str                  # forward_slowest, backward_peak, tp_comm_exposed
    value: Any
    evidence: list[str]
    confidence: float

@dataclass
class MainContradiction:
    code: str                  # ACTIVATION_MEMORY_BOUND
    layer: str                 # memory_strategy / parallelism / system_optimization
    reason: str
    supporting_facts: list[str]
    confidence: float

@dataclass
class PrioritizedAction:
    action_code: str           # enable_recompute
    layer: str
    priority: int
    expected_effect: str
    rationale: str
    anti_actions: list[str]    # 当前不建议优先做的事
```
```

这些对象最终可挂入 `AnalysisResult.details` 与 `AdvisorReport`。

### 7.2 新增技能链

建议在 `src/skills/v2/skills/analysis/` 或 `src/skills/v2/skills/diagnosis/` 下新增：

#### Skill 1: `step_attribution_skill.py`
职责：
- 读取 step 级 summary
- 判断最慢阶段与最痛阶段
- 输出 step 级 ObservationFacts

典型输出：
- `forward_slowest`
- `backward_slowest`
- `optimizer_step_slowest`
- `save_slowest`

#### Skill 2: `memory_attribution_skill.py`
职责：
- 读取内存指标与峰值
- 判断 `activation-bound` vs `state-bound`
- 输出 memory 级 ObservationFacts

典型输出：
- `forward_peak_high`
- `backward_peak_high`
- `step_peak_high`
- `activation_bound_likely`
- `state_bound_likely`

#### Skill 3: `communication_exposure_skill.py`
职责：
- 读取通信拆分、overlap、slow rank、bubble
- 判断哪类通信在暴露

典型输出：
- `tp_comm_exposed`
- `cp_comm_exposed`
- `ep_comm_exposed`
- `hsdp_comm_exposed`
- `pp_bubble_high`

#### Skill 4: `main_contradiction_skill.py`
职责：
- 读取前三个 skill 的 ObservationFacts
- 生成 `MainContradiction`
- 明确所属层

#### Skill 5: `action_prioritization_skill.py`
职责：
- 根据 `MainContradiction` 生成动作优先级
- 输出“先改什么，不先改什么”

---

## 8. 规则与提示词设计

### 8.1 规则层负责什么

规则层负责低歧义映射，例如：

- `recompute on/off` 敏感 + `forward/backward peak` 高 -> `ACTIVATION_MEMORY_BOUND`
- `optimizer.step` 最慢且 `step peak` 高 -> `STATE_MEMORY_BOUND`
- `bubble_ratio` 高 + `num_microbatches` 不足 -> `PIPELINE_BUBBLE_BOUND`
- `overlap_ratio` 低 + `comm_not_overlapped` 高 -> `COMMUNICATION_EXPOSED`

### 8.2 LLM 层负责什么

LLM 不应直接决定基础分类，主要负责：

- 把结构化结论组织成可读报告
- 补充解释和取舍理由
- 把动作写成更工程化的建议

### 8.3 Prompt 注入重点

需要把以下方法论注入 Prompt/Skill context：

```text
1. 先判断最慢 step 阶段
2. 再判断峰值显存属于哪类对象
3. 再判断哪类通信在暴露
4. 最后映射到三层：并行维 / 内存策略 / 系统优化
5. 一次优先只改最像主矛盾的一层
```

这应成为 Advisor 与相关诊断 Skill 的固定推理模板。

---

## 9. 与现有模块的集成方式

### 9.1 Orchestrator

在现有 agent 执行后，新增一段诊断技能链执行：

```text
TimelineAgent / MemoryAgent / CommunicationAgent
    -> Diagnostic Skill Chain
    -> AdvisorAgent
```

其中 Diagnostic Skill Chain 消费：

- `profiling_summary`
- `timeline_result.details`
- `memory_result.details`
- `communication_result.details`
- `root_cause_findings`（可选）

### 9.2 AdvisorAgent

AdvisorAgent 改成两阶段：

1. 结构化输入汇总
2. 基于诊断结果生成最终报告

新增输出板块建议：

- 观测结果摘要
- 主矛盾判断
- 优先调优动作
- 次优动作 / 当前不建议优先动作

### 9.3 RootCauseSkillEngine

建议不废弃，而是补成“规则库来源之一”：

- 保留现有 compare/root-cause 规则
- 新增单版本 step/memory/communication 规则文件
- 让 `main_contradiction_skill` 可以复用部分规则

即：

```text
RootCauseSkillEngine 继续做规则发现
Diagnosis Skills 负责跨维度整合
```

---

## 10. MVP 方案

### 10.1 本期最小落地范围

建议只做以下 4 件事：

1. 新增 `memory_attribution_skill`
2. 新增 `communication_exposure_skill`
3. 新增 `main_contradiction_skill`
4. 增强 `AdvisorAgent` 输出模板

先不做：
- 大规模重构 Agent
- 新的统一数据层
- 自动调参

### 10.2 MVP 能解决的问题

MVP 完成后，工具应至少能稳定输出：

```text
- 当前最慢的是 backward
- backward 峰值也最高
- 更像 activation + grad 叠峰
- 主矛盾属于内存策略层
- 优先动作是 recompute，暂不优先做 optimizer-state-offload
```

以及：

```text
- 当前暴露的是 HSDP grad communication
- overlap ratio 偏低，comm_not_overlapped 偏高
- 主矛盾属于系统优化层
- 优先动作是 grad fusion / overlap / wait 后移，不先改 CP
```

这就已经显著优于当前的泛化建议。

---

## 11. 后续扩展路线

### Phase 2

补强：

- `step_attribution_skill`
- `action_prioritization_skill`
- PP 专项诊断（bubble / stage imbalance / microbatch diagnosis）

### Phase 3

补强：

- 针对长序列 dense 的 `CP + recompute` 专项知识
- 针对 MoE 的 `EP + load balance + optimizer state` 专项知识
- 训练拓扑感知建议（TP 节点内、PP 节点间等）

### Phase 4

补强：

- 与 what-if 分析联动
- 自动实验建议生成
- 面向 PR/CI 的性能回归诊断模板

---

## 12. 风险与注意事项

### 12.1 风险一：事实不够完整

如果输入缺少：
- step 分阶段数据
- 峰值显存分阶段数据
- overlap 关键指标

则“主矛盾”判断会变弱。

处理方式：
- 允许 skill 输出 `confidence`
- 在报告中明确“结论基于当前可见数据”

### 12.2 风险二：LLM 覆盖结构化结果

如果最后让 LLM 重新自由组织，可能把规则链结论冲淡。

处理方式：
- 结构化 diagnosis 结果必须是最终报告的硬输入
- LLM 只能解释，不应推翻结构化判断

### 12.3 风险三：规则过早写死

第一版规则不要过细，不要一开始就绑定某个框架细节。

优先抽象到：
- step 阶段
- 对象类型
- 三层动作

---

## 13. 推荐实施顺序

### Step 1
新增设计文档（本文件）

### Step 2
实现 3 个 diagnosis skills：
- `memory_attribution_skill`
- `communication_exposure_skill`
- `main_contradiction_skill`

### Step 3
把 diagnosis skills 接到 Orchestrator / AdvisorAgent

### Step 4
补测试：
- activation-bound case
- state-bound case
- communication-exposed case
- pipeline-bubble case

### Step 5
根据真实 profiling 样例迭代规则

---

## 14. 最终建议

本项目当前最合适的演进路径不是“再加一个大模型顾问 Agent”，而是：

```text
先把 Timeline / Memory / Communication 三条结构化分析链，
提升为统一的“观测 -> 主矛盾 -> 优先动作”诊断链，
再让 Advisor 基于这个结构化结果生成最终报告。
```

这样做有三个直接收益：

1. 结果更稳定，可测试
2. 能把并行维/内存策略/系统优化知识真正沉淀进工具
3. 后续无论走 skills、subagents 还是 prompt 优化，都会有统一骨架可复用
