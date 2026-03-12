"""
Prompt Skills 模块

包含所有 LLM 推理指导技能
"""

from ..base_skill import PromptSkill, SkillCategory
from ..registry import register_skill


# ============================================================
# 诊断流程指导
# ============================================================

DIAGNOSIS_FLOW_PROMPT = PromptSkill(
    name="diagnosis_flow",
    display_name="性能诊断流程",
    description="指导 LLM 按照标准化流程进行性能诊断",
    category=SkillCategory.DIAGNOSIS,
    tags=["diagnosis", "flow", "guide"],
    prompt_template="""
# 性能诊断流程指南

## 诊断目标
分析当前训练任务的性能瓶颈，目标 MFU: {target_mfu}%

## 标准诊断流程

### 第一步：数据收集
1. 获取基本信息：
   - 模型参数量、Batch Size、Sequence Length
   - 硬件配置：NPU 型号、数量、互联拓扑
   - 并行策略：TP/PP/DP 配置

### 第二步：MFU 计算
调用 `calculate_mfu` 技能，获取当前 MFU

### 第三步：瓶颈定位
根据 MFU 结果，按以下顺序排查：

**如果 MFU < 20%（严重不足）**:
1. 检查 Host-Device 同步（Free Time）
2. 检查数据加载是否为瓶颈
3. 检查是否有大量小算子

**如果 MFU 20-40%（一般）**:
1. 调用 `check_overlap_ratio` 检查通信掩盖
2. 调用 `check_bandwidth_efficiency` 检查带宽
3. 调用 `detect_slow_rank` 检查慢卡

**如果 MFU 40-60%（良好）**:
1. 分析算子级别效率
2. 检查 Kernel Launch 开销
3. 优化内存访问模式

### 第四步：生成建议
基于诊断结果，生成具体可执行的优化建议

## 注意事项
- 每个结论必须有数据支撑
- 优化建议需考虑实施成本
- 关注投入产出比最高的优化点
"""
)


REPORT_FORMAT_PROMPT = PromptSkill(
    name="report_format",
    display_name="报告格式规范",
    description="指导 LLM 生成标准化的性能分析报告",
    category=SkillCategory.OPTIMIZATION,
    tags=["report", "format", "template"],
    prompt_template="""
# 性能分析报告格式规范

## 报告结构

### 1. 执行摘要 (Executive Summary)
- 一句话总结当前性能状态
- 关键指标：MFU、通信掩盖率、慢卡数量
- 主要瓶颈（1-2 个）

### 2. 环境信息
- 硬件配置表格
- 软件版本
- 并行策略配置

### 3. 性能分析

#### 3.1 计算效率
- MFU 数值及对比
- 主要计算密集型算子分析
- 算力利用率评估

#### 3.2 通信效率
- 通信掩盖率
- 各类集合操作占比
- 带宽利用率

#### 3.3 内存效率
- 显存使用峰值
- 内存碎片情况
- OOM 风险评估

### 4. 瓶颈诊断
使用以下格式描述每个瓶颈：

```
**瓶颈名称**: <描述>
- **影响程度**: 高/中/低
- **证据**: <具体数据>
- **根因分析**: <可能原因>
```

### 5. 优化建议
按优先级排序，每条建议包含：
- 具体操作步骤
- 预期收益
- 实施复杂度

### 6. 附录
- 详细数据表格
- 技能执行结果

## 语言风格
- 专业、准确、简洁
- 避免模糊表述（如"可能"、"也许"）
- 数据精确到合适的小数位
"""
)


OPTIMIZATION_STRATEGY_PROMPT = PromptSkill(
    name="optimization_strategy",
    display_name="优化策略指南",
    description="提供针对不同瓶颈的优化策略建议",
    category=SkillCategory.OPTIMIZATION,
    tags=["optimization", "strategy", "advice"],
    prompt_template="""
# 昇腾 NPU 训练优化策略库

## 计算瓶颈优化

### 如果算子效率低
1. **算子融合**: 合并相邻小算子，减少 Kernel Launch 开销
2. **Tiling 优化**: 调整算子的 Tiling 策略，提高 L2 Cache 命中率
3. **混合精度**: 使用 FP16/BF16 代替 FP32

### 如果 Kernel Launch 开销大
1. **CUDA Graph / NPU Graph**: 捕获多个 Kernel 为一个图，一次性下发
2. **减少同步点**: 检查不必要的 stream synchronize

## 通信瓶颈优化

### 如果通信掩盖率低
1. **调整通信时机**: 确保通信与计算交替进行
2. **梯度累积**: 增加 accumulation steps，减少通信频率
3. **Overlap 优化**: 使用 FSDP2 的改进掩盖策略

### 如果带宽利用率低
1. **检查 HCCL 配置**: 确认使用了最优的集合通信算法
2. **消息聚合**: 合并小消息，减少通信次数
3. **节点间通信优化**: 调整 TP/DP 比例，减少跨节点通信

## 内存瓶颈优化

### 如果显存不足
1. **激活值重计算**: 使用 gradient checkpointing
2. **ZeRO 优化**: 启用 ZeRO Stage 2/3
3. **序列并行**: 使用 Sequence Parallelism 减少激活值显存

### 如果内存碎片严重
1. **内存池**: 使用 PyTorch 的 memory allocator
2. **预分配**: 预先分配大块显存
3. **定期 GC**: 在合适的时机触发垃圾回收

## 并行策略优化

### 常见配置建议
- **7B 模型**: TP=1, PP=1, DP=8 (单机)
- **70B 模型**: TP=8, PP=4, DP=2 (4 机)
- **405B 模型**: TP=8, PP=8, DP=4+ (8+ 机)

### 调优原则
1. **TP 优先节点内**: TP 通信量大，应优先使用 HCCS
2. **PP 用于超大模型**: 当单卡放不下一层时使用 PP
3. **DP 用于扩展**: 数据量大时增加 DP

## 针对当前场景: {scenario}

{specific_advice}
"""
)


EXPERT_REASONING_PROMPT = PromptSkill(
    name="expert_reasoning",
    display_name="专家推理模式",
    description="引导 LLM 使用专家级推理分析问题",
    category=SkillCategory.DIAGNOSIS,
    tags=["reasoning", "expert", "analysis"],
    prompt_template="""
# 专家推理模式

作为一名资深的 AI Infra 工程师，请按照以下框架分析问题：

## 当前问题
{problem_description}

## 分析框架

### 1. 假设生成
基于现象，列出可能的原因（至少 3 个）：
- 假设 A: ...
- 假设 B: ...
- 假设 C: ...

### 2. 证据收集
对每个假设，调用相应的技能验证：
- 假设 A → 调用 `skill_name_a` 验证
- 假设 B → 调用 `skill_name_b` 验证
- 假设 C → 调用 `skill_name_c` 验证

### 3. 结论推导
基于技能返回的数据，得出结论：
- 数据支持的假设: ...
- 排除的假设: ...
- 需要进一步验证的假设: ...

### 4. 建议生成
基于确认的原因，给出具体建议

## 可用技能
{available_skills}

## 注意
- 每个结论必须有数据支撑
- 如果数据不足，明确说明需要补充什么信息
- 优先给出投入产出比最高的建议
"""
)


# ============================================================
# 注册所有 Prompt Skills
# ============================================================

def register_all_prompt_skills():
    """注册所有内置 Prompt 技能"""
    skills = [
        DIAGNOSIS_FLOW_PROMPT,
        REPORT_FORMAT_PROMPT,
        OPTIMIZATION_STRATEGY_PROMPT,
        EXPERT_REASONING_PROMPT,
    ]
    
    for skill in skills:
        register_skill(skill)
    
    return len(skills)


__all__ = [
    "DIAGNOSIS_FLOW_PROMPT",
    "REPORT_FORMAT_PROMPT",
    "OPTIMIZATION_STRATEGY_PROMPT",
    "EXPERT_REASONING_PROMPT",
    "register_all_prompt_skills",
]
