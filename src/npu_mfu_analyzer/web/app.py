"""
FastAPI 应用主入口

提供 RESTful API 和 WebSocket 接口。
"""

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("请安装 web 依赖: pip install fastapi uvicorn python-multipart websockets aiofiles")

from npu_mfu_analyzer.llm.llm_interface import LLMConfig
from npu_mfu_analyzer.agents.orchestrator import Orchestrator
from npu_mfu_analyzer.report.report_generator import ReportGenerator, ReportFormat

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="NPU MFU Analyzer API",
    description="昇腾 NPU 大模型训练性能分析工具 API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据目录
DATA_DIR = Path(os.environ.get("NPU_ANALYZER_DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"

# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 挂载静态文件
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 任务存储
tasks: Dict[str, "AnalysisTask"] = {}

# WebSocket 连接管理
class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
    
    async def send_progress(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()


# ==================== 数据模型 ====================

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisTask(BaseModel):
    """分析任务"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    profiling_path: str = ""
    report_path: Optional[str] = None
    error: Optional[str] = None
    
    class Config:
        use_enum_values = True


class AnalyzeRequest(BaseModel):
    """分析请求"""
    profiling_path: str = Field(..., description="Profiling 数据路径")
    llm_backend: str = Field(default="mock", description="LLM 后端 (mock/openai/claude)")
    output_format: str = Field(default="html", description="输出格式 (markdown/html/json)")


class CompareRequest(BaseModel):
    """对比分析请求"""
    path_a: str = Field(..., description="基准 Profiling 数据路径")
    path_b: str = Field(..., description="当前 Profiling 数据路径")
    label_a: str = Field(default="基准版本 (A)", description="版本 A 标签")
    label_b: str = Field(default="当前版本 (B)", description="版本 B 标签")
    llm_backend: str = Field(default="mock", description="LLM 后端 (mock/openai/claude)")
    output_format: str = Field(default="html", description="输出格式 (markdown/html)")
    force: bool = Field(default=False, description="跳过相似度检查，强制对比")


class AnalyzeResponse(BaseModel):
    """分析响应"""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task: AnalysisTask
    report_url: Optional[str] = None


# ==================== API 路由 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """首页 - 返回 Web UI"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    # 降级返回简单页面
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NPU MFU Analyzer</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #1a73e8; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🚀 NPU MFU Analyzer API</h1>
        <p>昇腾 NPU 大模型训练性能分析工具</p>
        
        <h2>API 端点</h2>
        <div class="endpoint"><strong>POST /api/analyze</strong> - 启动分析任务</div>
        <div class="endpoint"><strong>GET /api/tasks/{task_id}</strong> - 查询任务状态</div>
        <div class="endpoint"><strong>GET /api/reports/{task_id}</strong> - 获取分析报告</div>
        <div class="endpoint"><strong>WS /ws/{task_id}</strong> - WebSocket 实时进度</div>
        
        <p>📚 <a href="/docs">Swagger 文档</a> | <a href="/redoc">ReDoc 文档</a></p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/upload", response_model=dict)
async def upload_profiling(file: UploadFile = File(...)):
    """
    上传 Profiling 数据文件
    
    支持格式：
    - .tar.gz / .tgz
    - .zip
    - .rar
    """
    # 生成唯一 ID
    upload_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / upload_id
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    file_path = upload_path / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    
    # TODO: 自动解压
    
    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "path": str(upload_path),
        "message": "上传成功，请使用 /api/analyze 启动分析"
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def start_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    启动分析任务
    
    分析会在后台运行，可通过 WebSocket 或轮询获取进度。
    """
    # 验证路径
    profiling_path = Path(request.profiling_path)
    if not profiling_path.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在: {request.profiling_path}")
    
    # 创建任务
    task = AnalysisTask(
        profiling_path=request.profiling_path,
        message="任务已创建，等待执行"
    )
    tasks[task.id] = task
    
    # 后台执行分析
    background_tasks.add_task(
        run_analysis_task,
        task.id,
        request.profiling_path,
        request.llm_backend,
        request.output_format
    )
    
    return AnalyzeResponse(
        task_id=task.id,
        status=task.status,
        message="分析任务已启动"
    )


async def run_analysis_task(
    task_id: str,
    profiling_path: str,
    llm_backend: str,
    output_format: str
):
    """后台执行分析任务"""
    task = tasks.get(task_id)
    if not task:
        return
    
    try:
        # 更新状态
        task.status = TaskStatus.RUNNING
        task.progress = 10
        task.message = "正在加载数据..."
        await manager.send_progress(task_id, {"progress": 10, "message": task.message})
        
        # 创建 Orchestrator
        llm_config = LLMConfig(backend=llm_backend)
        orchestrator = Orchestrator(profiling_path, llm_config=llm_config)
        
        task.progress = 20
        task.message = "正在分析 Timeline..."
        await manager.send_progress(task_id, {"progress": 20, "message": task.message})
        
        # 执行分析
        format_map = {
            "markdown": ReportFormat.MARKDOWN,
            "html": ReportFormat.HTML,
            "json": ReportFormat.JSON,
        }
        report_format = format_map.get(output_format, ReportFormat.HTML)
        
        report = await orchestrator.run(output_format=report_format)
        
        task.progress = 80
        task.message = "正在生成报告..."
        await manager.send_progress(task_id, {"progress": 80, "message": task.message})
        
        # 保存报告
        report_filename = f"{task_id}.{output_format}"
        report_path = REPORT_DIR / report_filename
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.final_report)
        
        # 完成
        task.status = TaskStatus.COMPLETED
        task.progress = 100
        task.message = "分析完成"
        task.completed_at = datetime.now().isoformat()
        task.report_path = str(report_path)
        
        await manager.send_progress(task_id, {
            "progress": 100,
            "message": "分析完成",
            "status": "completed",
            "report_url": f"/api/reports/{task_id}"
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"分析失败: {e}"
        
        await manager.send_progress(task_id, {
            "progress": 0,
            "message": task.message,
            "status": "failed",
            "error": str(e)
        })


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    report_url = None
    if task.status == TaskStatus.COMPLETED:
        report_url = f"/api/reports/{task_id}"
    
    return TaskStatusResponse(task=task, report_url=report_url)


@app.get("/api/tasks")
async def list_tasks():
    """列出所有任务"""
    return {"tasks": list(tasks.values())}


@app.get("/api/reports/{task_id}")
async def get_report(task_id: str):
    """获取分析报告"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"任务未完成: {task.status}")
    
    if not task.report_path or not Path(task.report_path).exists():
        raise HTTPException(status_code=404, detail="报告文件不存在")
    
    # 根据文件类型返回
    report_path = Path(task.report_path)
    if report_path.suffix == ".html":
        return FileResponse(report_path, media_type="text/html")
    elif report_path.suffix == ".json":
        return FileResponse(report_path, media_type="application/json")
    else:
        return FileResponse(report_path, media_type="text/markdown")


@app.post("/api/compare", response_model=AnalyzeResponse)
async def start_comparison(request: CompareRequest, background_tasks: BackgroundTasks):
    """
    启动对比分析任务

    对比两个 Profiling 数据的差异，分析性能变化的根本原因。
    """
    # 验证路径
    path_a = Path(request.path_a)
    path_b = Path(request.path_b)

    if not path_a.exists():
        raise HTTPException(status_code=400, detail=f"路径 A 不存在: {request.path_a}")
    if not path_b.exists():
        raise HTTPException(status_code=400, detail=f"路径 B 不存在: {request.path_b}")

    # 创建任务
    task = AnalysisTask(
        profiling_path=f"{request.path_a} vs {request.path_b}",
        message="对比分析任务已创建，等待执行"
    )
    tasks[task.id] = task

    # 后台执行对比
    background_tasks.add_task(
        run_comparison_task,
        task.id,
        request.path_a,
        request.path_b,
        request.label_a,
        request.label_b,
        request.llm_backend,
        request.output_format,
        request.force,
    )

    return AnalyzeResponse(
        task_id=task.id,
        status=task.status,
        message="对比分析任务已启动"
    )


async def run_comparison_task(
    task_id: str,
    path_a: str,
    path_b: str,
    label_a: str,
    label_b: str,
    llm_backend: str,
    output_format: str,
    force: bool,
):
    """后台执行对比分析任务"""
    task = tasks.get(task_id)
    if not task:
        return

    try:
        from npu_mfu_analyzer.analyzers.comparison_orchestrator import ComparisonOrchestrator

        # 更新状态
        task.status = TaskStatus.RUNNING
        task.progress = 10
        task.message = "正在加载 Profiling 数据..."
        await manager.send_progress(task_id, {"progress": 10, "message": task.message})

        # 创建 Orchestrator
        llm_config = LLMConfig(backend=llm_backend)
        orchestrator = ComparisonOrchestrator(
            path_a=path_a,
            path_b=path_b,
            label_a=label_a,
            label_b=label_b,
            llm_config=llm_config,
            force=force,
        )

        task.progress = 30
        task.message = "正在进行相似度检测和差异分析..."
        await manager.send_progress(task_id, {"progress": 30, "message": task.message})

        # 确定输出格式
        format_map = {
            "markdown": ReportFormat.MARKDOWN,
            "html": ReportFormat.HTML,
        }
        report_format = format_map.get(output_format, ReportFormat.HTML)

        # 执行对比
        report = await orchestrator.run(output_format=report_format)

        task.progress = 80
        task.message = "正在生成对比报告..."
        await manager.send_progress(task_id, {"progress": 80, "message": task.message})

        # 保存报告
        ext = "html" if output_format == "html" else "md"
        report_filename = f"{task_id}.{ext}"
        report_path = REPORT_DIR / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.final_report)

        if report.success:
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.message = f"对比分析完成: {report.summary}"
            task.completed_at = datetime.now().isoformat()
            task.report_path = str(report_path)

            await manager.send_progress(task_id, {
                "progress": 100,
                "message": task.message,
                "status": "completed",
                "report_url": f"/api/reports/{task_id}",
            })
        else:
            # 不可比或其他失败
            if report.error == "NOT_COMPARABLE":
                task.status = TaskStatus.COMPLETED  # 仍视为完成，有报告
                task.progress = 100
                task.message = f"对比分析完成（不建议对比）: {report.summary}"
                task.completed_at = datetime.now().isoformat()
                task.report_path = str(report_path)

                await manager.send_progress(task_id, {
                    "progress": 100,
                    "message": task.message,
                    "status": "completed",
                    "report_url": f"/api/reports/{task_id}",
                    "warning": "NOT_COMPARABLE",
                })
            else:
                task.status = TaskStatus.FAILED
                task.error = report.error
                task.message = f"对比分析失败: {report.error}"
                await manager.send_progress(task_id, {
                    "progress": 0,
                    "message": task.message,
                    "status": "failed",
                    "error": report.error,
                })

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"对比分析失败: {e}"

        await manager.send_progress(task_id, {
            "progress": 0,
            "message": task.message,
            "status": "failed",
            "error": str(e),
        })


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks.pop(task_id)
    
    # 删除报告文件
    if task.report_path and Path(task.report_path).exists():
        Path(task.report_path).unlink()
    
    return {"message": "任务已删除", "task_id": task_id}


# ==================== WebSocket ====================

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket 实时进度推送
    
    连接后会实时接收任务进度更新。
    """
    await manager.connect(websocket, task_id)
    
    try:
        # 发送当前状态
        task = tasks.get(task_id)
        if task:
            await websocket.send_json({
                "progress": task.progress,
                "message": task.message,
                "status": task.status
            })
        
        # 保持连接
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # 可以处理客户端消息
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)


# ==================== 启动入口 ====================

def create_app():
    """创建应用实例"""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
