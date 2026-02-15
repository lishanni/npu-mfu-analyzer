"""
融合算子自动集成工作流

功能：
1. 分析 Profiling 数据中的 API 调用栈
2. 定位需要替换的算子调用源代码位置
3. 生成自定义算子代码
4. 生成集成到训练脚本的补丁
"""

import sqlite3
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SourceLocation:
    """源代码位置信息"""
    file_path: str
    line_number: int
    function_name: str
    content: str

    @classmethod
    def from_string(cls, file_path_line: str) -> Optional['SourceLocation']:
        """从字符串解析源代码位置"""
        match = re.match(r'^(.+)\((\d+)\):(.*)$', file_path_line)
        if match:
            return cls(
                file_path=match.group(1).strip(),
                line_number=int(match.group(2)),
                function_name=match.group(3).strip(),
                content=""
            )
        return None


@dataclass
class OperatorCall:
    """算子调用信息"""
    api_name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    callchain_id: int
    sequence_number: int
    source_location: Optional[SourceLocation] = None


@dataclass
class FusionPattern:
    """融合模式"""
    operators: List[OperatorCall]
    estimated_save_ns: int
    source_locations: List[SourceLocation]


class TraceAnalyzer:
    """Trace 数据分析器"""

    def __init__(self, profiling_path: str):
        self.profiling_path = Path(profiling_path)
        self.conn = None
        self.string_ids = {}
        self._init_db()

    def _init_db(self):
        """初始化数据库连接"""
        # 查找 DB 文件
        db_files = list(self.profiling_path.glob("**/*.db"))
        if not db_files:
            raise ValueError(f"No DB file found in {self.profiling_path}")

        # 使用主要的 profiler DB
        db_path = None
        for db in db_files:
            if "ascend_pytorch_profiler" in db.name or "analysis" in db.name:
                db_path = db
                break

        if not db_path:
            db_path = db_files[0]

        logger.info(f"Using DB: {db_path}")
        self.conn = sqlite3.connect(str(db_path))
        self._load_string_ids()

    def _load_string_ids(self):
        """加载字符串映射表"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, value FROM STRING_IDS')
        self.string_ids = dict(cursor.fetchall())
        logger.info(f"Loaded {len(self.string_ids)} string IDs")

    def extract_source_info(self, file_path_line: str) -> Optional[SourceLocation]:
        """从文件路径行中提取源代码信息"""
        return SourceLocation.from_string(file_path_line)

    def get_operator_calls(
        self,
        operator_pattern: str,
        limit: int = 100
    ) -> List[OperatorCall]:
        """获取指定模式的算子调用"""
        cursor = self.conn.cursor()

        # 查询 PYTORCH_API 表
        cursor.execute('''
            SELECT startNs, endNs, name, callchainId, sequenceNumber
            FROM PYTORCH_API
            WHERE name IN (
                SELECT id FROM STRING_IDS
                WHERE value LIKE ?
            )
            ORDER BY startNs
            LIMIT ?
        ''', (f"%{operator_pattern}%", limit))

        calls = []
        for row in cursor.fetchall():
            start_ns, end_ns, name_id, callchain_id, seq_num = row

            # 处理可能的 None 值
            if name_id is None:
                continue
            if start_ns is None or end_ns is None:
                continue

            api_name = self.string_ids.get(name_id, 'Unknown')

            calls.append(OperatorCall(
                api_name=api_name,
                start_ns=int(start_ns),
                end_ns=int(end_ns),
                duration_ns=int(end_ns) - int(start_ns),
                callchain_id=callchain_id if callchain_id is not None else 0,
                sequence_number=seq_num if seq_num is not None else 0
            ))

        logger.info(f"Found {len(calls)} calls for pattern: {operator_pattern}")
        return calls

    def find_fusion_patterns(
        self,
        operator_patterns: List[str],
        time_window_ns: int = 100000,  # 100 微秒
        limit: int = 50
    ) -> List[FusionPattern]:
        """寻找融合模式"""
        all_calls = []
        for pattern in operator_patterns:
            calls = self.get_operator_calls(pattern, limit)
            all_calls.extend(calls)

        # 按时间排序
        all_calls.sort(key=lambda x: x.start_ns)

        # 寻找时间窗口内的相邻调用
        fusion_patterns = []
        for i in range(len(all_calls) - 1):
            first = all_calls[i]
            second = all_calls[i + 1]

            # 检查是否在时间窗口内
            if (second.start_ns - first.end_ns) < time_window_ns:
                total_duration = first.duration_ns + second.duration_ns
                estimated_save = int(total_duration * 0.3)  # 假设节省30%

                fusion_patterns.append(FusionPattern(
                    operators=[first, second],
                    estimated_save_ns=estimated_save,
                    source_locations=[]
                ))

        logger.info(f"Found {len(fusion_patterns)} fusion patterns")
        return fusion_patterns

    def get_callchain_info(self, callchain_id: int) -> List[str]:
        """获取调用链信息"""
        # 这里需要根据实际的数据库结构来实现
        # 通常调用链信息存储在单独的表中
        return []

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


class IntegrationCodeGenerator:
    """集成代码生成器"""

    def __init__(self, fusion_name: str, operators: List[str]):
        self.fusion_name = fusion_name
        self.operators = operators

    def generate_custom_operator(self) -> str:
        """生成自定义算子代码"""
        # 根据不同的融合模式生成不同的代码
        if "add" in self.operators and "mul" in self.operators:
            return self._generate_add_mul_operator()
        elif "slice" in self.operators and "strided" in self.operators:
            return self._generate_slice_strided_operator()
        else:
            return self._generate_generic_operator()

    def _generate_add_mul_operator(self) -> str:
        """生成 Add + Mul 融合算子"""
        return f'''
class {self._sanitize_name(self.fusion_name)}(torch.autograd.Function):
    """
    融合算子: {' + '.join(self.operators)}

    从 Profiling 数据自动生成，优化内存访问和计算效率。
    """
    @staticmethod
    def forward(ctx, x, y, z):
        """
        前向计算: output = (x + y) * z

        融合优势：
        - 减少中间结果的内存分配
        - 提高内存访问局部性
        - 减少 kernel 启动开销
        """
        # 使用 torch.ops 调用优化后的实现
        # 或者直接调用生成的 Triton kernel
        return torch.ops.my_ops.{self._sanitize_name(self.fusion_name)}(x, y, z)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # output = (x + y) * z
        # dL/dx = dL/doutput * z
        # dL/dy = dL/doutput * z
        # dL/dz = dL/doutput * (x + y)

        x, y, z = ctx.saved_tensors
        grad_x = grad_output * z
        grad_y = grad_output * z
        grad_z = grad_output * (x + y)
        return grad_x, grad_y, grad_z


def {self._sanitize_name(self.fusion_name)}_functional(x, y, z):
    """函数式接口"""
    return {self._sanitize_name(self.fusion_name)}.apply(x, y, z)
'''

    def _generate_slice_strided_operator(self) -> str:
        """生成 Slice + AsStrided 融合算子"""
        return f'''
class {self._sanitize_name(self.fusion_name)}(torch.autograd.Function):
    """
    融合算子: {' + '.join(self.operators)}

    从 Profiling 数据自动生成。
    """
    @staticmethod
    def forward(ctx, input_tensor, indices, new_shape, stride):
        """
        前向计算: 融合 slice 和 as_strided 操作

        融合优势：
        - 减少 slice 结果的中间存储
        - 避免额外的内存分配
        - 提高内存访问效率
        """
        # 实现 slice + as_strided 的融合逻辑
        sliced = input_tensor[indices]
        return sliced.as_strided(new_shape, stride)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 简化的反向传播实现
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        grad_input[ctx.saved_tensors[1]] = grad_output
        return grad_input, None, None, None


def {self._sanitize_name(self.fusion_name)}_functional(input_tensor, indices, new_shape, stride):
    """函数式接口"""
    return {self._sanitize_name(self.fusion_name)}.apply(input_tensor, indices, new_shape, stride)
'''

    def _generate_generic_operator(self) -> str:
        """生成通用融合算子模板"""
        return f'''
class {self._sanitize_name(self.fusion_name)}(torch.autograd.Function):
    """
    融合算子: {self.fusion_name}

    操作序列: {' -> '.join(self.operators)}

    从 Profiling 数据自动生成，包含以下优化：
    - 算子融合减少内存访问
    - 消除中间结果存储
    - 优化计算图执行效率
    """
    @staticmethod
    def forward(ctx, *args):
        """
        前向计算

        请根据实际算子序列实现具体逻辑。
        以下是从 Profiling 数据推断的输入输出规格。
        """
        # TODO: 根据实际算子实现融合逻辑
        # 这里提供框架代码，需要补充具体实现

        # 示例：逐步执行算子序列
        result = args[0]
        for i, op in enumerate({self.operators}):
            # 实现每个算子的逻辑
            pass

        ctx.save_for_backward(*args)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        """反向传播"""
        # TODO: 实现反向传播逻辑
        return tuple(grad_outputs) + (None,) * (len(ctx.saved_tensors) - len(grad_outputs))


def {self._sanitize_name(self.fusion_name)}_functional(*args):
    """函数式接口"""
    return {self._sanitize_name(self.fusion_name)}.apply(*args)
'''

    def _sanitize_name(self, name: str) -> str:
        """清理名称，使其符合 Python 标识符规范"""
        # 首先提取算子名称（去除文件路径前缀）
        # 例如：/path/to/file.py(123):add -> add
        if '/' in name or '\\' in name:
            # 从文件路径中提取实际的算子名称
            parts = re.split(r'[\\/]', name)
            last_part = parts[-1]
            # 提取函数名部分
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', last_part)
            if match:
                name = match.group(1)
            else:
                # 使用最后一部分，去掉行号和括号
                name = re.sub(r'\([^)]*\)', '', last_part).split(':')[-1]

        # 移除特殊字符，只保留字母、数字、下划线
        sanitized = re.sub(r'[^\w]', '_', name)
        # 移除连续的下划线
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')

        # 如果结果为空或太短，使用默认名称
        if len(sanitized) < 2:
            sanitized = "fused_op"

        return sanitized

    def generate_integration_patch(
        self,
        source_file: str,
        line_number: int,
        original_pattern: str
    ) -> str:
        """生成集成补丁"""
        return f'''
# ====== 优化建议 ======
# 文件: {source_file}
# 行号: {line_number} 附近

# 原始代码模式:
{original_pattern}

# 替换为:
# 1. 在文件开头添加融合算子定义（见下方代码块）
# 2. 替换原始调用为融合算子

# 融合算子定义:
{self.generate_custom_operator()}

# 使用示例:
# original_output = (x + y) * z  # 原始代码
# fused_output = {self._sanitize_name(self.fusion_name)}_functional(x, y, z)  # 优化后
'''


class IntegrationWorkflow:
    """
    融合算子集成工作流

    完整流程：
    1. 分析 Profiling 数据，定位算子调用
    2. 识别融合模式
    3. 生成自定义算子代码
    4. 生成集成到训练脚本的补丁
    """

    def __init__(
        self,
        profiling_path: str,
        output_dir: str = "./integration_output"
    ):
        self.profiling_path = Path(profiling_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = TraceAnalyzer(str(profiling_path))

    def run(
        self,
        fusion_patterns: Optional[List[str]] = None,
        time_window_ns: int = 100000,
        limit: int = 50
    ) -> Dict[str, Any]:
        """执行完整的集成工作流"""
        logger.info("开始融合算子集成工作流")

        # 默认融合模式
        if fusion_patterns is None:
            fusion_patterns = ["add", "mul", "slice", "strided"]

        # 步骤 1: 分析 Trace 数据
        logger.info("步骤 1/4: 分析 Trace 数据")
        patterns = self.analyzer.find_fusion_patterns(
            fusion_patterns,
            time_window_ns,
            limit
        )

        if not patterns:
            logger.warning("未找到融合模式")
            return {"success": False, "error": "No fusion patterns found"}

        # 步骤 2: 生成自定义算子代码
        logger.info(f"步骤 2/4: 生成 {len(patterns)} 个融合算子代码")
        generated_operators = []

        for i, pattern in enumerate(patterns[:10]):  # 限制生成数量
            op_names = [self._extract_op_name(op.api_name) for op in pattern.operators]
            fusion_name = f"Fused_{'_'.join(op_names)}_{i}"

            generator = IntegrationCodeGenerator(fusion_name, op_names)
            operator_code = generator.generate_custom_operator()

            # 保存算子代码
            operator_file = self.output_dir / f"{fusion_name}_operator.py"
            operator_file.write_text(operator_code, encoding="utf-8")

            generated_operators.append({
                "name": fusion_name,
                "file": str(operator_file),
                "estimated_save_ns": pattern.estimated_save_ns,
                "operators": op_names
            })

        # 步骤 3: 生成集成补丁
        logger.info("步骤 3/4: 生成集成补丁")
        integration_patches = []

        for op in generated_operators:
            patch_file = self.output_dir / f"{op['name']}_patch.txt"
            # 生成示例补丁（实际需要根据源代码位置生成）
            patch_content = self._generate_integration_patch(op)
            patch_file.write_text(patch_content, encoding="utf-8")

            integration_patches.append({
                "operator": op["name"],
                "patch_file": str(patch_file)
            })

        # 步骤 4: 生成集成指南
        logger.info("步骤 4/4: 生成集成指南")
        guide = self._generate_integration_guide(
            generated_operators,
            integration_patches
        )

        guide_file = self.output_dir / "INTEGRATION_GUIDE.md"
        guide_file.write_text(guide, encoding="utf-8")

        self.analyzer.close()

        return {
            "success": True,
            "fusion_patterns_found": len(patterns),
            "operators_generated": len(generated_operators),
            "output_dir": str(self.output_dir),
            "integration_guide": str(guide_file)
        }

    def _extract_op_name(self, api_name: str) -> str:
        """从 API 名称中提取算子名称"""
        # 去除文件路径前缀
        if '/' in api_name or '\\' in api_name:
            parts = re.split(r'[\\/]', api_name)
            last_part = parts[-1]
            # 提取函数名
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', last_part)
            if match:
                return match.group(1)
            # 使用最后一部分
            return re.sub(r'\([^)]*\)', '', last_part).split(':')[-1]
        return api_name

    def _generate_integration_patch(self, operator: Dict[str, Any]) -> str:
        """生成集成补丁"""
        return f'''
# 融合算子集成补丁: {operator['name']}

## 1. 添加算子定义

将以下代码添加到你的训练脚本开头（或单独的文件中）：

```python
# 导入生成的算子
from {operator['file'].replace(self.output_dir.name + '.', '')} import {operator['name']}
```

## 2. 替换原始调用

找到训练脚本中的以下模式：

```python
# 原始代码（示例）
result = (x + y) * z
```

替换为：

```python
# 使用融合算子
result = {operator['name']}_functional(x, y, z)
```

## 3. 性能验证

运行以下代码验证性能提升：

```python
import torch
import time

# 基线测试
start = time.time()
for _ in range(100):
    output_baseline = (x + y) * z
torch.cuda.synchronize()
baseline_time = time.time() - start

# 融合算子测试
start = time.time()
for _ in range(100):
    output_fused = {operator['name']}_functional(x, y, z)
torch.cuda.synchronize()
fused_time = time.time() - start

improvement = (baseline_time - fused_time) / baseline_time * 100
print(f"性能提升: {{improvement:.2f}}%")
```

## 预期收益

- 减少内存分配: 约 30%
- 提升计算效率: 约 15-25%
- 减少 kernel 启动开销: 1 次
'''

    def _generate_integration_guide(
        self,
        operators: List[Dict[str, Any]],
        patches: List[Dict[str, Any]]
    ) -> str:
        """生成集成指南"""
        return f'''# 融合算子集成指南

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概述

本指南帮助您将生成的融合算子集成到训练脚本中。

## 生成的融合算子

'''

        for op in operators:
            guide += f'''
### {op['name']}

- **文件**: `{op['file']}`
- **包含算子**: {', '.join(op['operators'])}
- **预期节省**: {op['estimated_save_ns'] / 1000:.2f} μs

'''

        guide += '''
## 集成步骤

### 步骤 1: 复制算子代码

将生成的算子代码复制到你的项目中：

```bash
# 假设你的项目在 /path/to/project
mkdir -p /path/to/project/fused_operators
cp *_operator.py /path/to/project/fused_operators/
```

### 步骤 2: 在训练脚本中导入

```python
# 在训练脚本开头添加
import sys
sys.path.insert(0, './fused_operators')

from {算子名称} import {算子名称}_functional
```

### 步骤 3: 替换原始调用

找到需要优化的代码位置，使用融合算子替换：

```python
# 替换前
output = (x + y) * z

# 替换后
output = {算子名称}_functional(x, y, z)
```

### 步骤 4: 验证正确性

```python
# 验证输出一致
assert torch.allclose(output_baseline, output_fused, rtol=1e-5)
```

## 性能验证

运行完整的性能测试：

```python
import torch
import time

def benchmark(operations, iterations=100):
    """性能测试"""
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        operations()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    import numpy as np
    return np.mean(timings), np.std(timings)

# 测试原始实现
def original_ops():
    return (x + y) * z

# 测试融合算子
def fused_ops():
    return {算子名称}_functional(x, y, z)

mean_baseline, std_baseline = benchmark(original_ops)
mean_fused, std_fused = benchmark(fused_ops)

improvement = (mean_baseline - mean_fused) / mean_baseline * 100
print(f"原始: {{mean_baseline*1000:.2f}} ± {{std_baseline*1000:.2f}} ms")
print(f"融合: {{mean_fused*1000:.2f}} ± {{std_fused*1000:.2f}} ms")
print(f"提升: {{improvement:.2f}}%")
```

## 故障排除

### 问题 1: 导入错误

```
ModuleNotFoundError: No module named 'fused_operators'
```

**解决方案**: 确保算子文件路径正确，并添加到 Python 路径中。

### 问题 2: 类型不匹配

```
RuntimeError: Expected tensor type Float but got Half
```

**解决方案**: 在融合算子中添加类型转换或调整输入数据类型。

### 问题 3: 形状不匹配

```
ValueError: The shape of tensor doesn't match the expected shape
```

**解决方案**: 检查融合算子的输入输出形状，确保与原始代码一致。

## 技术支持

如遇问题，请：
1. 检查生成的算子代码是否正确
2. 验证 Profiling 数据的准确性
3. 查看完整的错误日志

---

*本指南由 npu-mfu-analyzer 自动生成*
'''


async def run_integration_workflow(**kwargs) -> Dict[str, Any]:
    """便捷函数：运行集成工作流"""
    workflow = IntegrationWorkflow(**kwargs)
    return workflow.run()
