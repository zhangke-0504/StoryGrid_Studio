import logging
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

def setup_logging() -> None:
    """最简日志配置，避免重复处理器。"""
    root = logging.getLogger()
    if root.handlers:
        return  # 已配置
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root.addHandler(handler)

def create_app() -> FastAPI:
    """创建 FastAPI 实例并挂载中间件。"""
    app = FastAPI(
        title="Manuscript API",
        description="简化版后端服务（可扩展路由）",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 开放跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

def register_routers(app: FastAPI) -> None:
    """集中注册路由，保持可扩展结构。"""
    api_base = "/api"  

    # working_flow 路由
    from routers.story_grid.interface import router as story_grid_router
    app.include_router(story_grid_router, prefix=f"{api_base}/story_grid", tags=["story_grid"])


# 实例化应用并注册路由
setup_logging()
app = create_app()
register_routers(app)
def mount_static(app: FastAPI) -> None:
    # Serve frontend static files. When running from source, static files are
    # located at backend/dist/www; when frozen (exe in backend/dist), static
    # files are next to the exe in backend/dist/www -> sys.executable parent.
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        static_dir = os.path.join(exe_dir, "www")
    else:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join(backend_dir, "dist", "www")

    # Only mount if the directory exists
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    else:
        logging.getLogger("app").warning("Static directory not found: %s", static_dir)

mount_static(app)