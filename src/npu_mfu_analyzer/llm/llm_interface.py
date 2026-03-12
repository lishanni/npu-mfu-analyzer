"""
LLM 接口层 - 统一的大语言模型访问接口

支持多种后端：OpenAI, Claude, 本地模型等。
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """消息对象"""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMConfig:
    """LLM 配置"""
    backend: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self):
        # 从环境变量获取 API Key
        if self.api_key is None:
            if self.backend == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.backend == "claude":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

        # 从环境变量获取 base_url
        if self.base_url is None:
            if self.backend == "claude":
                self.base_url = os.getenv("ANTHROPIC_BASE_URL")


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Any = None


class LLMInterface(ABC):
    """LLM 接口基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def complete(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """
        发送消息并获取响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            LLMResponse
        """
        pass
    
    async def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        简化的对话接口
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示（可选）
            
        Returns:
            响应文本
        """
        messages = []
        
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        messages.append(Message(role="user", content=prompt))
        
        response = await self.complete(messages)
        return response.content


class OpenAIBackend(LLMInterface):
    """OpenAI 后端"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                
                kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        return self._client
    
    async def complete(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """调用 OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=[m.to_dict() for m in messages],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class ClaudeBackend(LLMInterface):
    """Anthropic Claude 后端"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """延迟初始化 Claude 客户端"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                # 支持自定义 base_url（用于代理服务）
                client_kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    client_kwargs["base_url"] = self.config.base_url

                self._client = AsyncAnthropic(**client_kwargs)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        return self._client
    
    async def complete(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """调用 Claude API"""
        try:
            # 分离 system message
            system_content = ""
            chat_messages = []
            
            for msg in messages:
                if msg.role == "system":
                    system_content = msg.content
                else:
                    chat_messages.append(msg.to_dict())
            
            response = await self.client.messages.create(
                model=kwargs.get("model", self.config.model),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                system=system_content,
                messages=chat_messages,
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                raw_response=response,
            )
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class MockBackend(LLMInterface):
    """Mock 后端，用于测试"""

    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """返回 Mock 响应"""
        # 检查是否是 AIKG 代码生成请求
        prompt = messages[0].content if messages else ""

        if "Fusion Operator Generation Request" in prompt or "Triton" in prompt:
            # 返回模拟的 Triton 代码
            mock_triton_code = '''```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_flash_attention(
    q_ptr, k_ptr, v_ptr, output_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride.vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N, M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention 融合算子实现

    融合了以下操作:
    1. Q @ K^T (注意力分数计算)
    2. Softmax (归一化)
    3. @ V (加权聚合)

    使用 Tiling 技术减少 HBM 访问，提升性能。
    """
    # 获取程序 ID
    pid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # 计算 block 范围
    m_start = pid_m * BLOCK_M
    m_end = min(m_start + BLOCK_M, M)

    # 加载 Q 块
    q_offsets = (
        m_start[:, None] * stride_qm +
        tl.arange(0, BLOCK_M)[None, :] * stride_qm
    )
    q_block = tl.load(q_ptr + q_offsets)

    # 计算 Q @ K^T
    # ... 省略详细实现

    # Softmax
    # ... 省略详细实现

    # 计算 @ V
    # ... 省略详细实现

    # 存储结果
    output_offsets = (
        m_start[:, None] * stride_om +
        tl.arange(0, BLOCK_M)[None, :] * stride_om
    )
    tl.store(output_ptr + output_offsets, output_block)


def flash_attention(q, k, v):
    """
    Flash Attention 前端接口

    Args:
        q: Query tensor [B, H, M, K]
        k: Key tensor [B, H, N, K]
        v: Value tensor [B, H, N, K]

    Returns:
        output: Attention output [B, H, M, K]
    """
    B, H, M, K = q.shape
    N = k.shape[2]

    # 配置 block 大小
    BLOCK_M = 128
    BLOCK_N = 128

    # 输出 tensor
    output = torch.empty_like(q)

    # 启动 kernel
    grid = (1, H, triton.cdiv(M, BLOCK_M))
    fused_flash_attention[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, N, M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output
```

**性能特点:**
- 使用 Tiling 技术减少 HBM 访问
- 融合 Q@K^T、Softmax、@V 三个操作
- 适用于昇腾 NPU (Triton-Ascend 后端)
- 预期性能提升: 5x (相比标准 Attention)

**编译命令:**
```bash
python -c "import torch; from flash_attention_impl import flash_attention; print('OK')"
```
'''
            return LLMResponse(
                content=mock_triton_code,
                model="mock-triton",
                usage={"prompt_tokens": 500, "completion_tokens": 800, "total_tokens": 1300},
            )
        else:
            # 默认简单响应
            return LLMResponse(
                content="[Mock Response] This is a test response for development.",
                model="mock",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            )


class OllamaBackend(LLMInterface):
    """
    Ollama 本地 LLM 后端
    
    Ollama 是免费的本地 LLM 服务，支持 Llama, Qwen, DeepSeek 等模型。
    安装: https://ollama.ai
    
    使用方法:
        1. 安装 Ollama: brew install ollama (Mac) 或下载安装包
        2. 启动服务: ollama serve
        3. 下载模型: ollama pull qwen2.5:7b
        4. 配置: backend=ollama, model=qwen2.5:7b
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Ollama 默认地址
        self.base_url = config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # 默认模型
        if config.model == "gpt-4-turbo-preview":
            config.model = "qwen2.5:7b"  # 默认使用 Qwen
    
    async def complete(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """调用 Ollama API"""
        import aiohttp
        
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    return LLMResponse(
                        content=data.get("message", {}).get("content", ""),
                        model=data.get("model", self.config.model),
                        usage={
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                        },
                        raw_response=data,
                    )
                    
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise Exception(f"无法连接到 Ollama 服务 ({self.base_url})。请确保 Ollama 已启动: ollama serve")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class DeepSeekBackend(LLMInterface):
    """
    DeepSeek API 后端
    
    DeepSeek 提供高性价比的 API 服务。
    注册: https://platform.deepseek.com
    
    使用方法:
        export DEEPSEEK_API_KEY="your-key"
        backend=deepseek
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = config.base_url or "https://api.deepseek.com/v1"
        # 默认模型
        if config.model == "gpt-4-turbo-preview":
            config.model = "deepseek-chat"
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    async def complete(
        self, 
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """调用 DeepSeek API（OpenAI 兼容）"""
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=[m.to_dict() for m in messages],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                raw_response=response,
            )
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise


class LLMFactory:
    """LLM 工厂类"""
    
    _backends = {
        "openai": OpenAIBackend,
        "claude": ClaudeBackend,
        "mock": MockBackend,
        "ollama": OllamaBackend,
        "deepseek": DeepSeekBackend,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLMInterface:
        """
        创建 LLM 实例
        
        Args:
            config: LLM 配置
            
        Returns:
            LLMInterface 实例
        """
        backend_class = cls._backends.get(config.backend)
        
        if backend_class is None:
            raise ValueError(f"Unknown backend: {config.backend}. Available: {list(cls._backends.keys())}")
        
        return backend_class(config)
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        """注册自定义后端"""
        cls._backends[name] = backend_class
