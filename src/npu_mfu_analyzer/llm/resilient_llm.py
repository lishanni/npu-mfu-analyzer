"""
弹性 LLM 客户端

提供重试、降级、超时处理等容错机制。
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from npu_mfu_analyzer.llm.llm_interface import (
    LLMInterface, LLMConfig, LLMResponse, Message, LLMFactory
)

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 30.0  # 最大延迟（秒）
    exponential_base: float = 2.0  # 指数退避基数


@dataclass
class FallbackConfig:
    """降级配置"""
    enabled: bool = True
    fallback_backends: List[str] = field(default_factory=lambda: ["ollama", "mock"])


@dataclass
class TimeoutConfig:
    """超时配置"""
    request_timeout: float = 120.0  # 单次请求超时（秒）
    total_timeout: float = 300.0  # 总超时（秒）


@dataclass
class ResilientConfig:
    """弹性配置"""
    retry: RetryConfig = field(default_factory=RetryConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)


class ResilientLLM(LLMInterface):
    """
    弹性 LLM 客户端
    
    功能：
    1. 自动重试（指数退避）
    2. 超时处理
    3. 后端降级（主后端失败时自动切换）
    4. 错误统计
    
    Usage:
        config = LLMConfig(backend="openai")
        resilient_config = ResilientConfig()
        llm = ResilientLLM(config, resilient_config)
        response = await llm.complete(messages)
    """
    
    def __init__(
        self,
        config: LLMConfig,
        resilient_config: Optional[ResilientConfig] = None
    ):
        super().__init__(config)
        self.resilient_config = resilient_config or ResilientConfig()
        
        # 主后端
        self._primary_backend: Optional[LLMInterface] = None
        
        # 降级后端列表
        self._fallback_backends: List[LLMInterface] = []
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retries": 0,
            "fallbacks": 0,
            "timeouts": 0,
            "total_latency_ms": 0,
        }
    
    @property
    def primary_backend(self) -> LLMInterface:
        """延迟初始化主后端"""
        if self._primary_backend is None:
            self._primary_backend = LLMFactory.create(self.config)
        return self._primary_backend
    
    def _get_fallback_backends(self) -> List[LLMInterface]:
        """获取降级后端列表"""
        if not self._fallback_backends and self.resilient_config.fallback.enabled:
            for backend_name in self.resilient_config.fallback.fallback_backends:
                if backend_name != self.config.backend:
                    try:
                        fallback_config = LLMConfig(backend=backend_name)
                        backend = LLMFactory.create(fallback_config)
                        self._fallback_backends.append(backend)
                        logger.debug(f"Registered fallback backend: {backend_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create fallback backend {backend_name}: {e}")
        return self._fallback_backends
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """
        发送请求（带重试和降级）
        """
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # 1. 尝试主后端（带重试）
            response = await self._call_with_retry(
                self.primary_backend,
                messages,
                **kwargs
            )
            self.stats["successful_requests"] += 1
            return response
            
        except Exception as primary_error:
            logger.warning(f"Primary backend failed: {primary_error}")
            
            # 2. 尝试降级后端
            if self.resilient_config.fallback.enabled:
                for fallback in self._get_fallback_backends():
                    try:
                        logger.info(f"Trying fallback backend: {type(fallback).__name__}")
                        self.stats["fallbacks"] += 1
                        
                        response = await self._call_with_retry(
                            fallback,
                            messages,
                            **kwargs
                        )
                        self.stats["successful_requests"] += 1
                        return response
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback backend failed: {fallback_error}")
                        continue
            
            # 3. 所有后端都失败
            self.stats["failed_requests"] += 1
            raise Exception(f"All LLM backends failed. Primary error: {primary_error}")
            
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_latency_ms"] += elapsed_ms
    
    async def _call_with_retry(
        self,
        backend: LLMInterface,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """带重试的调用"""
        retry_config = self.resilient_config.retry
        timeout_config = self.resilient_config.timeout
        
        last_error = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                # 带超时的调用
                response = await asyncio.wait_for(
                    backend.complete(messages, **kwargs),
                    timeout=timeout_config.request_timeout
                )
                return response
                
            except asyncio.TimeoutError:
                self.stats["timeouts"] += 1
                last_error = Exception(f"Request timeout after {timeout_config.request_timeout}s")
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            
            # 重试前等待（指数退避）
            if attempt < retry_config.max_retries:
                self.stats["retries"] += 1
                delay = min(
                    retry_config.base_delay * (retry_config.exponential_base ** attempt),
                    retry_config.max_delay
                )
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        raise last_error or Exception("Unknown error")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0
            stats["avg_latency_ms"] = 0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retries": 0,
            "fallbacks": 0,
            "timeouts": 0,
            "total_latency_ms": 0,
        }


class LLMPool:
    """
    LLM 连接池
    
    管理多个 LLM 后端，支持负载均衡和健康检查。
    """
    
    def __init__(self, configs: List[LLMConfig]):
        """
        Args:
            configs: 多个 LLM 配置
        """
        self.backends: List[LLMInterface] = []
        self.health_status: Dict[int, bool] = {}
        
        for config in configs:
            try:
                backend = LLMFactory.create(config)
                self.backends.append(backend)
                self.health_status[len(self.backends) - 1] = True
            except Exception as e:
                logger.warning(f"Failed to create backend {config.backend}: {e}")
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """轮询调用健康的后端"""
        for idx, backend in enumerate(self.backends):
            if not self.health_status.get(idx, False):
                continue
            
            try:
                response = await backend.complete(messages, **kwargs)
                return response
            except Exception as e:
                logger.warning(f"Backend {idx} failed: {e}")
                self.health_status[idx] = False
        
        raise Exception("No healthy backends available")
    
    async def health_check(self):
        """健康检查"""
        test_messages = [Message(role="user", content="test")]
        
        for idx, backend in enumerate(self.backends):
            try:
                await asyncio.wait_for(
                    backend.complete(test_messages),
                    timeout=10.0
                )
                self.health_status[idx] = True
            except Exception:
                self.health_status[idx] = False
