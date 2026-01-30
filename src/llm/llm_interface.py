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
                
                self._client = AsyncAnthropic(api_key=self.config.api_key)
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
        return LLMResponse(
            content="[Mock Response] This is a test response for development.",
            model="mock",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )


class LLMFactory:
    """LLM 工厂类"""
    
    _backends = {
        "openai": OpenAIBackend,
        "claude": ClaudeBackend,
        "mock": MockBackend,
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
