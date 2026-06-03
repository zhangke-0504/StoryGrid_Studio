import os
import argparse
import logging
import uvicorn
from get_app import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI 启动参数")
    parser.add_argument("-p", "--port", type=int, default=8890, help="启动端口号")
    parser.add_argument("-H", "--host", type=str, default="127.0.0.1", help="绑定地址")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--env", type=str, choices=["dev", "test", "release"], default="dev", help="运行环境")
    args = parser.parse_args()

    # 基础环境变量
    os.environ["APP_ENV"] = args.env
    os.environ["APP_PORT"] = str(args.port)

    setup_logging()
    logger = logging.getLogger("app")
    logger.info(f"Starting API on {args.host}:{args.port} (env={args.env}, reload={args.reload})")

    try:
        # 使用字符串引用，便于 --reload 下的自动重载
        uvicorn.run(
            "get_app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正常退出。")