# pip install esdk-obs-python pyyaml --trusted-host pypi.org
import os
import yaml
import asyncio
import concurrent.futures
import requests
import traceback
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, Union, Callable
from pathlib import Path
from obs import GetObjectHeader, PutObjectHeader, ObsClient


import logging
logger = logging.getLogger("tracker_logger")

class OBSConfig:
    """OBS配置管理类"""
    
    def __init__(self, config_path: str = "config/obs/config.yaml"):
        self.config_path = config_path
        self.domestic_config = None
        self.oversea_config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """从YAML文件加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            # 国内配置
            self.domestic_config = {
                'access_key_id': config_data.get('ACCESS_KEY'),
                'secret_access_key': config_data.get('SECRET_ACCESS_KEY'),
                'server': config_data.get('SERVER'),
                'bucket_name': 'laihuaresources'
            }
            
            # 海外配置 - 从环境变量或配置文件中获取
            self.oversea_config = {
                'access_key_id': config_data.get('OVERSEA_ACCESS_KEY', 'RCHH6MOBUEIKAXCBACFD'),
                'secret_access_key': config_data.get('OVERSEA_SECRET_KEY', 'zR1ferjkO9SXOqHSOm8oap0KLh1zjExGcP85Bc5W'),
                'server': config_data.get('OVERSEA_SERVER', 'obs.ap-southeast-3.myhuaweicloud.com'),
                'bucket_name': 'global-resources'
            }
                
        except FileNotFoundError:
            raise Exception(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"配置文件格式错误: {e}")
        except Exception as e:
            raise Exception(f"加载配置失败: {e}")


class OBSManager:
    """华为云OBS管理类"""
    
    def __init__(self, config_path: str = "config/obs/config.yaml"):
        self.config = OBSConfig(config_path)
        self.domestic_client = None
        self.oversea_client = None
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """初始化OBS客户端"""
        try:
            # 初始化国内客户端
            self.domestic_client = ObsClient(
                access_key_id=self.config.domestic_config['access_key_id'],
                secret_access_key=self.config.domestic_config['secret_access_key'],
                server=self.config.domestic_config['server']
            )
            
            # 初始化海外客户端
            self.oversea_client = ObsClient(
                access_key_id=self.config.oversea_config['access_key_id'],
                secret_access_key=self.config.oversea_config['secret_access_key'],
                server=self.config.oversea_config['server']
            )
            
        except Exception as e:
            raise Exception(f"初始化OBS客户端失败: {e}")
    
    def _get_client_and_bucket(self, oversea: bool = False) -> tuple:
        """根据区域获取对应的客户端和桶名"""
        if oversea:
            return self.oversea_client, self.config.oversea_config['bucket_name']
        else:
            return self.domestic_client, self.config.domestic_config['bucket_name']
    
    def _get_content_type(self, filename: str) -> str:
        """根据文件扩展名获取Content-Type"""
        extension = Path(filename).suffix.lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mp3': 'audio/mp3',
            '.wav': 'audio/wav',
            '.png': 'image/png',
            '.jpg': 'image/jpg',
            '.jpeg': 'image/jpeg',
            '.zip': 'application/zip',
            '.html': 'text/html',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain'
        }
        return content_types.get(extension, 'application/octet-stream')
    
    def _get_obs_host(self, oversea: bool = False) -> str:
        """获取OBS主机地址"""
        if oversea:
            return "https://fresource.laihua.com"
        else:
            return "https://glrs.laihua.com"
    
    def download(self, object_key: str, download_path: str, oversea: bool = False, use_buffer: bool = False) -> any:
        """
        从OBS下载文件
        
        Args:
            object_key: 对象键，格式如 "2023-12-1/5099555a-005f-40e3-9970-2ddbc0ee2cce.mp3"
            download_path: 本地下载路径
            oversea: 是否使用海外节点
            use_buffer: 是否使用内存缓冲区下载
            
        Returns:
            bool: 下载是否成功
        """
        try:
            client, bucket_name = self._get_client_and_bucket(oversea)
            headers = GetObjectHeader()
            
            resp = client.getObject(bucket_name, object_key, download_path, headers=headers, loadStreamInMemory=use_buffer)
            
            if resp.status < 300:
                if use_buffer:
                    return resp
                return True
            else:
                logger.info(f'下载失败 - requestId: {resp.requestId}, errorCode: {resp.errorCode}, errorMessage: {resp.errorMessage}')
                return False
                
        except Exception as e:
            logger.info(f'下载异常: {e}')
            return False
    
    async def download_async(self, object_key: str, download_path: str, oversea: bool = False, use_buffer: bool = False) -> any:
        """异步下载文件"""
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool, self.download, object_key, download_path, oversea, use_buffer)
        return result
    
    # ------------------- 上传（含进度回调） -------------------
    def upload(self, local_path: str, object_key: str = None, oversea: bool = False,
               progress_hook: Optional[Callable[[int, int, float], None]] = None) -> Optional[str]:
        """
        上传文件到OBS（同步）
        
        Args:
            local_path: 本地文件路径
            object_key: 对象键（可选，如为None则自动生成）
            oversea: 是否使用海外节点
            progress_hook: 可选回调，签名为 (transferred_bytes, total_bytes, elapsed_seconds) -> None
            
        Returns:
            str: 文件URL，失败返回None
        """
        try:
            if not os.path.exists(local_path):
                logger.info(f'文件不存在: {local_path}')
                return None
            
            # 自动生成object_key
            if object_key is None:
                filename = Path(local_path).name
                formatted_date = date.today().strftime("%Y-%m-%d")
                object_key = f"{formatted_date}/{filename}"
            
            client, bucket_name = self._get_client_and_bucket(oversea)
            headers = PutObjectHeader()
            headers.contentType = self._get_content_type(object_key)
            headers.storageClass = "WARM"

            # OBS SDK 要求的 progress callback 参数形式：
            # progressCallback(transferredAmount, totalAmount, totalSeconds)
            progress_wrapper = None
            if progress_hook is not None:
                def progress_wrapper(transferredAmount, totalAmount, totalSeconds):
                    try:
                        # 直接把 SDK 的参数传给外层 hook（字节数、字节数、秒）
                        progress_hook(int(transferredAmount), int(totalAmount), float(totalSeconds))
                    except Exception:
                        # 回调异常不能影响上传
                        logger.info("progress_hook raised exception:\n", traceback.format_exc())

            # 调用 putFile（传入 progressCallback 即可）
            if progress_wrapper is not None:
                resp = client.putFile(bucket_name, object_key, local_path, headers=headers, progressCallback=progress_wrapper)
            else:
                resp = client.putFile(bucket_name, object_key, local_path, headers=headers)
            
            if resp.status < 300:
                file_url = f"{self._get_obs_host(oversea)}/{object_key}"
                
                # 验证文件可访问
                try:
                    validation_resp = requests.get(file_url, timeout=5)
                    if validation_resp.status_code == 200:
                        return file_url
                    else:
                        logger.info(f"文件验证失败，状态码: {validation_resp.status_code}")
                        return file_url  # 仍然返回URL，但记录警告
                except requests.RequestException as e:
                    logger.info(f"文件验证请求异常: {e}")
                    return file_url  # 仍然返回URL，但记录警告
            else:
                logger.info(f'上传失败 - errorCode: {resp.errorCode}, errorMessage: {resp.errorMessage}')
                return None
                
        except Exception as e:
            logger.info(f'上传异常: {e}')
            return None
    

    async def upload_async(self, local_path: str, object_key: str = None, oversea: bool = False,
                           progress_hook: Optional[Callable[[int, int, float], None]] = None) -> Optional[str]:
        """异步上传文件（在线程池中执行同步 upload，可传入 progress_hook）"""
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool, lambda: self.upload(local_path, object_key, oversea, progress_hook))
        return result
    
    
    def fetch(self, object_key: str, oversea: bool = False) -> Optional[bytes]:
        """
        获取文件内容（适用于小文件）
        
        Args:
            object_key: 对象键
            oversea: 是否使用海外节点
            
        Returns:
            bytes: 文件内容，失败返回None
        """
        try:
            client, bucket_name = self._get_client_and_bucket(oversea)
            headers = GetObjectHeader()
            
            resp = client.getObject(bucket_name, object_key, loadStreamInMemory=True, headers=headers)
            
            if resp.status < 300:
                return resp.body.buffer
            else:
                logger.info(f'获取文件失败 - errorCode: {resp.errorCode}, errorMessage: {resp.errorMessage}')
                return None
                
        except Exception as e:
            logger.info(f'获取文件异常: {e}')
            return None
    
    async def fetch_async(self, object_key: str, oversea: bool = False) -> Optional[bytes]:
        """异步获取文件内容"""
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool, self.fetch, object_key, oversea)
        return result
    
    def create_signature(self, file_name: str, content_type: str = None) -> Optional[Dict[str, str]]:
        """
        创建表单上传签名
        
        Args:
            file_name: 文件名
            content_type: 内容类型，如为None则自动判断
            
        Returns:
            Dict: 包含签名信息的字典
        """
        try:
            if content_type is None:
                content_type = self._get_content_type(file_name)
            
            client, bucket_name = self._get_client_and_bucket(False)  # 默认使用国内
            expires = 3600
            form_params = {'x-obs-acl': 'public-read', 'content-type': content_type}
            
            resp = client.createPostSignature(bucket_name, file_name, expires, form_params)
            
            return {
                "originPolicy": resp.originPolicy,
                "policy": resp.policy,
                "signature": resp.signature,
            }
                
        except Exception as e:
            logger.info(f'创建签名异常: {e}')
            return None
    
    def close(self):
        """关闭OBS客户端连接"""
        if self.domestic_client:
            self.domestic_client.close()
        if self.oversea_client:
            self.oversea_client.close()


# 环境配置类（保留原有逻辑）
class EnvConfig:
    """环境配置类"""
    
    def __init__(self, env_mode: str = None, platform: str = None, region: str = None):
        self.env_mode = env_mode or os.getenv("ENV", "debug")
        self.platform = platform or os.getenv("PLATFORM", "web")
        self.region = region or os.getenv("REGION", "overseas")
        self.is_debug = "debug" in self.env_mode
        self.is_mobile = "mobile" in self.platform
        self.is_overseas = "overseas" in self.region
        
    @property
    def is_production(self) -> bool:
        return not self.is_debug


class HostConfig:
    """主机配置类"""
    
    def __init__(self, env_config: EnvConfig):
        self.env_config = env_config

    @property
    def obs_host(self) -> str:
        """获取对象存储主机地址"""
        if self.env_config.is_overseas:
            return "https://fresource.laihua.com"
        else:
            return "https://glrs.laihua.com"


# 全局实例
env_config = EnvConfig()
host_config = HostConfig(env_config)


# ------------------- 示例与测试用 __main__ -------------------
def progress_hook(transferred: int, total: int, elapsed_seconds: float):
    try:
        kb_per_s = (transferred / 1024.0) / (elapsed_seconds if elapsed_seconds > 0 else 1.0)
        percent = (transferred * 100.0 / total) if total > 0 else 0.0
        logger.info(f"[progress] {transferred}/{total} bytes ({percent:.2f}%), {kb_per_s:.2f} KB/s, elapsed: {elapsed_seconds:.2f}s")
    except Exception:
        logger.info(f"progress hook error:{traceback.format_exc()}\n")


async def main():
    """测试用例（演示普通上传 + 带进度的异步上传）"""
    obs_manager = OBSManager("config/obs/config.yaml")
    try:
        # test_file = "tmp/xxxxxxxxxxx/normalized/whole_xxxxxxxxxxx1.mp4"  # 请修改为实际文件
        # test_file = "TestingCode/static/rag_demo_by_zk.mp4"  # 请修改为实际文件
        test_file = "TestingCode/static/冉冰.jpeg"  # 请修改为实际文件
        # test_file = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if not os.path.exists(test_file):
            print("测试文件不存在，请先生成拼接文件后再运行此脚本。")
            return

        # print("=== 测试：默认上传（upload_async） ===")
        # try:
        #     url = await obs_manager.upload_async(test_file, oversea=False)
        #     if url:
        #         print("默认上传成功:", url)
        #     else:
        #         print("默认上传返回 None")
        # except Exception as e:
        #     print("默认上传失败:", repr(e))

        print("\n=== 测试：带进度的异步上传（upload_with_progress_async） ===")
        try:
            # object_key 可选，设为 None 则自动以日期/文件名生成
            url = await obs_manager.upload_async(
                test_file,
                object_key=None,
                oversea=True,
                progress_hook=progress_hook
            )
            if url:
                print("带进度上传成功:", url)
            else:
                print("带进度上传返回 None")
        except Exception as e:
            print("带进度上传失败:", repr(e))

    finally:
        obs_manager.close()


# async def main():
#     """测试用例"""
#     # 初始化OBS管理器
#     obs_manager = OBSManager("config/obs/config.yaml")
#     try:
#         # 测试文件上传
#         test_file = "tmp/xxxxxxxxxxx/normalized/whole_xxxxxxxxxxx.mp4"  # 请确保此文件存在或修改为实际文件路径
        
#         if os.path.exists(test_file):
#             print("开始上传测试文件...")
#             file_url = await obs_manager.upload_async(test_file, oversea=False)
            
#             if file_url:
#                 print(f"文件上传成功: {file_url}")
#         else:
#             print("路径不存在")
#     except Exception as e:
#         print(f"obs上传出错, Error: {repr(e)}")
                
#     # # 测试文件下载
#     # try:
#     #     download_path = "tmp/downloaded_test_image.jpg"
#     #     success = await obs_manager.download_async(
#     #         "2025-10-27/韩立紫灵.jpg",  # 请根据实际上传的object_key修改
#     #         download_path, 
#     #         oversea=False
#     #     )
        
#     #     if success:
#     #         print(f"文件下载成功: {download_path}")
#     #         # 清理测试文件
#     #         os.remove(download_path)
#     #     else:
#     #         print("文件下载失败")
#     # except Exception as e:
#     #     print(f"obs文件下载失败, Error: {repr(e)}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())