#!/bin/bash
# Ollama 安装和配置脚本

set -e

echo "=========================================="
echo "NPU MFU Analyzer - Ollama 配置脚本"
echo "=========================================="

# 检查 Ollama 是否已安装
if command -v ollama &> /dev/null; then
    echo "✓ Ollama 已安装"
    ollama --version
else
    echo "安装 Ollama..."
    
    # macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ollama
    # Linux
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "不支持的操作系统: $OSTYPE"
        exit 1
    fi
fi

# 启动 Ollama 服务
echo ""
echo "启动 Ollama 服务..."
ollama serve &
sleep 3

# 下载推荐模型
echo ""
echo "下载推荐模型..."

# Qwen2.5 - 中文能力强，适合分析中文报告
echo "下载 qwen2.5:7b..."
ollama pull qwen2.5:7b

# 可选：更小的模型用于快速测试
# echo "下载 qwen2.5:3b..."
# ollama pull qwen2.5:3b

# 验证
echo ""
echo "验证安装..."
ollama list

echo ""
echo "=========================================="
echo "Ollama 配置完成!"
echo ""
echo "使用方法:"
echo "  1. 确保 Ollama 服务运行: ollama serve"
echo "  2. 在 config/settings.yaml 中配置:"
echo "     llm:"
echo "       backend: ollama"
echo "       model: qwen2.5:7b"
echo "=========================================="
