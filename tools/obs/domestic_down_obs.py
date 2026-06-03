import asyncio
import time
import logging
import os
from typing import Optional, Dict, Any
import requests
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_task")


class DownloadTaskClient:
    """
    Create a remote download task then poll until finished or error.
    """
    def __init__(self, base_url: Optional[str] = None, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_vinabot', 'config.yaml')
            )

        self.config_path = os.path.normpath(config_path)
        self._config = self._load_config()

        resolved_base_url = base_url or self._resolve_api_url_by_env()
        self.create_url, self.check_url = self._build_download_urls(resolved_base_url)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return data
        except FileNotFoundError:
            logger.warning("config.yaml not found at %s, falling back to default download url", self.config_path)
            return {}
        except Exception as e:
            logger.exception("failed to load config.yaml: %s", e)
            return {}

    def _resolve_api_url_by_env(self) -> str:
        env = (os.getenv("APP_ENV", "test") or "test").strip().lower()
        is_release = env == "release"

        if is_release:
            resolved = self._config.get("download_api_base_url_release")
        else:
            resolved = self._config.get("download_api_base_url_test")

        if not resolved:
            resolved = "http://api.vinabot.net/apicenter/release/api/download/create" if is_release else "http://api.vinabot.net/apicenter/test/api/download/create"

        logger.info("vinabot download api env=%s, using create url=%s", env, resolved)
        return resolved

    def _build_download_urls(self, configured_url: str) -> tuple[str, str]:
        normalized_url = configured_url.rstrip("/")

        if normalized_url.endswith("/download/create"):
            create_url = normalized_url
            check_url = f"{normalized_url[:-len('/create')]}/check"
        elif normalized_url.endswith("/download/check"):
            check_url = normalized_url
            create_url = f"{normalized_url[:-len('/check')]}/create"
        else:
            base_url = normalized_url
            create_url = f"{base_url}/download/create"
            check_url = f"{base_url}/download/check"

        return create_url, check_url

    async def create_and_wait(
        self,
        source_url: str,
        overseas: bool = False,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Create a download task and poll the check endpoint until status is 'done' or 'error'.

        Returns the "data" object from the check endpoint on completion.
        Raises on create failure or timeout.
        """
        loop = asyncio.get_running_loop()

        payload = {"url": source_url, "overseas": overseas}

        # create
        try:
            resp = await loop.run_in_executor(None, lambda: requests.post(self.create_url, json=payload, timeout=15))
        except Exception as e:
            raise RuntimeError(f"create request failed: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"create returned HTTP {resp.status_code}: {resp.text}")

        try:
            j = resp.json()
        except Exception as e:
            raise RuntimeError(f"create returned invalid JSON: {e}")

        task_id = j.get("task_id") or j.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"create response missing task_id: {j}")

        logger.info("created task_id=%s", task_id)

        # poll
        deadline = time.time() + timeout
        last_status = None
        while time.time() < deadline:
            try:
                resp2 = await loop.run_in_executor(None, lambda: requests.get(self.check_url, params={"task_id": task_id}, timeout=15))
            except Exception as e:
                logger.info("check request error: %s", e)
                await asyncio.sleep(poll_interval)
                continue

            if resp2.status_code != 200:
                logger.info("check HTTP %s: %s", resp2.status_code, resp2.text)
                await asyncio.sleep(poll_interval)
                continue

            try:
                j2 = resp2.json()
            except Exception as e:
                logger.info("check returned invalid JSON: %s", e)
                await asyncio.sleep(poll_interval)
                continue

            data = j2.get("data") or {}
            status = data.get("status")
            if status != last_status:
                logger.info("task %s status=%s", task_id, status)
                last_status = status

            if status == "done":
                return data
            if status == "error":
                # return the data anyway so caller can inspect error fields
                return data

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Polling timed out after {timeout} seconds for task {task_id}")


async def main():
    client = DownloadTaskClient()
    # example source URL from prompt
    source_url = """
https://ark-project.tos-cn-beijing.volces.com/doc_image/r2v_tea_pic2.jpg
"""

    try:
        result = await client.create_and_wait(source_url, overseas=False, poll_interval=1.0, timeout=300.0)
        print("final result:", result)
    except Exception as e:
        print("error:", repr(e))


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())