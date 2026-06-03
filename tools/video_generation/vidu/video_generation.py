import os
import time
import yaml
import asyncio
import logging
import aiohttp  # 用aiohttp替换httpx
from pathlib import Path
from typing import Optional

from tools.obs.domestic_down_obs import DownloadTaskClient
from tools.obs.obs_files import OBSManager


logger = logging.getLogger(__name__)

class ViduVideoGenerator:
    def __init__(self, config_path="config/vidu/config.yaml"):
        """初始化，自动从 YAML 配置文件加载 API_KEY"""
        self.config_info = self._load_config(config_path)
        # self.model = self.config_info['MODEL']
        self.model = "viduq2"
        self.download_dir = Path("tmp/download")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.config_info['API_KEY']}"
        }

    def _load_config(self, path):
        """从 YAML 配置文件读取 API_KEY"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "API_KEY" not in config:
            raise KeyError("配置文件中缺少 API_KEY 字段")

        return config

    async def refer_gen_video(self, subjects, prompt, resolution="720p", 
                        duration=10, audio=True, aspect_ratio="16:9", model="viduq3-beta"):
        """
        参考生视频（异步），最新支持的模型是viduq3-beta
        """
        url = "https://api.vidu.cn/ent/v2/reference2video"
        payload = {
            "model": model or self.model,
            "subjects": subjects,
            "prompt": prompt,
            "duration": duration,
            "audio": audio,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio

        }
        print("请求vidu用到的payload:", payload)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["task_id"]

    async def start_end_gen_video(self, image_urls, prompt, duration=5, seed=0,
                       resolution="1080p", movement_amplitude="auto"):
        """
        首尾帧生成任务（异步）
        """
        url = "https://api.vidu.cn/ent/v2/start-end2video"
        payload = {
            "model": "viduq2-pro",
            "images": image_urls,
            "prompt": prompt,
            "duration": duration,
            "seed": seed,
            "resolution": resolution,
            "movement_amplitude": movement_amplitude,
            "watermark": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["task_id"]

    async def img2video(self, images, prompt, audio=True, voice_id="",
                        duration=5, seed=0, resolution="1080p", movement_amplitude="auto", off_peak=False):
        """
        图生视频异步任务
        """
        url = "https://api.vidu.cn/ent/v2/img2video"
        payload = {
            "model": "viduq3-pro",
            "images": images,
            "prompt": prompt,
            "audio": audio,
            "voice_id": voice_id,
            "duration": duration,
            "seed": seed,
            "resolution": resolution,
            "movement_amplitude": movement_amplitude,
            "off_peak": off_peak
        }
        print("请求vidu图生视频payload:", payload)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["task_id"]

    async def check_status(self, task_id):
        """查询任务状态（异步）"""
        url = f"https://api.vidu.cn//ent/v2/tasks/{task_id}/creations"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()

    async def download_video(self, url: str, target_path: Path) -> Path:
        """下载生成的视频到本地 tmp 目录。"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(target_path, "wb") as file_obj:
                    async for chunk in response.content.iter_chunked(8192):
                        file_obj.write(chunk)
        return target_path

    async def upload_to_obs(self, file_path: Path, oversea: bool = True) -> Optional[str]:
        """上传文件到 OBS 并返回访问链接。"""
        obs_manager = OBSManager("config/obs/config.yaml")
        try:
            return await obs_manager.upload_async(str(file_path), oversea=oversea)
        finally:
            obs_manager.close()

    async def cleanup_file(self, file_path: Path) -> None:
        """删除本地临时文件。"""
        if file_path.exists():
            file_path.unlink()

    async def wait_for_task_and_upload_to_obs(
        self,
        task_id: str,
        poll_interval: int = 10,
        oversea: bool = True,
        filename_prefix: str = "vidu_video",
    ) -> Optional[str]:
        """轮询视频任务，成功后上传到 OBS 并返回 OBS 链接。"""
        while True:
            status_data = await self.check_status(task_id)
            current_state = status_data["state"]
            print(f"🔄 任务状态: {current_state} (轮询时间: {time.strftime('%Y-%m-%d %H:%M:%S')})")

            if current_state == "success":
                video_url = status_data["creations"][0]["url"]
                print(f"供应商video_url: {video_url}")
                obs_url = await self.download_and_upload_to_obs(
                    video_url,
                    filename_prefix=filename_prefix,
                    oversea=oversea,
                )
                print(f"🌐 OBS 链接: {obs_url}")
                return obs_url

            if current_state == "failed":
                print("生成失败的时候，返回的结构是：", status_data)
                err_code = status_data.get("err_code", "未知错误")
                print(f"❌ 任务失败! 错误代码: {err_code}")
                return None

            await asyncio.sleep(poll_interval)

    async def download_and_upload_to_obs(
        self,
        video_url: str,
        filename_prefix: str = "vidu_video",
        oversea: bool = True,
    ) -> str:
        """下载供应商视频、上传到 OBS，并在完成后删除本地临时文件。"""
        local_path = self.download_dir / f"{filename_prefix}_{int(time.time())}.mp4"
        try:
            downloader = DownloadTaskClient()
            try:
                task_result = await downloader.create_and_wait(
                    source_url=video_url,
                    overseas=oversea,
                    poll_interval=1.0,
                    timeout=300.0,
                )
            except Exception as e:
                logger.error("远端创建下载任务失败，video_url=%s, err=%s", video_url, e)
                raise

            source_download_url = task_result.get("obs_url") or task_result.get("url")
            if not source_download_url:
                logger.error("下载任务返回结果中缺少可下载地址，task_result=%s", task_result)
                raise RuntimeError("没有可下载的 URL")

            logger.info("远端下载完成，开始下载到本地: %s", source_download_url)

            await self.download_video(source_download_url, local_path)
            print(f"🎉 视频生成成功，已下载到: {local_path}")

            obs_url = await self.upload_to_obs(local_path, oversea=oversea)
            if not obs_url:
                raise RuntimeError("OBS 上传失败")

            return obs_url
        finally:
            if local_path.exists():
                await self.cleanup_file(local_path)
                print(f"🧹 已清理本地文件: {local_path}")
            print(f"🧼 临时文件是否已删除: {not local_path.exists()}")


if __name__ == "__main__":
    async def main():
        generator = ViduVideoGenerator()
        obs_url = None

        # # ================ 测试图生视频 ================
        # print("================== 开始测试图生视频 =================")
        # images = [
        #     "https://fresource.laihua.com/2026-01-17/韩立.jpg"
        # ]
        # prompt = "男子沉思许久，抬头望天，留下了眼泪，说道：“既生瑜，何生亮”."
        # try:
        #     task_id = await generator.img2video(
        #         images, prompt, audio=True, voice_id="",
        #         duration=10, seed=0, resolution="720p", movement_amplitude="auto", off_peak=False
        #     )
        #     print(f"✅ 任务创建成功! Task ID: {task_id}")

        #     while True:
        #         status_data = await generator.check_status(task_id)
        #         current_state = status_data["state"]
        #         print(f"🔄 任务状态: {current_state} (轮询时间: {time.strftime('%Y-%m-%d %H:%M:%S')})")

        #         if current_state == "success":
        #             video_url = status_data["creations"][0]["url"]
        #             print(f"\n🎉 视频生成成功! 下载地址: {video_url}")
        #             break
        #         elif current_state == "failed":
        #             print("生成失败时候返回的结构：", status_data)
        #             err_code = status_data.get("err_code", "未知错误")
        #             print(f"❌ 任务失败! 错误代码: {err_code}")
        #             break
        #         await asyncio.sleep(10)
        # except Exception as e:
        #     print(f"⚠️ 发生异常: {repr(e)}")

#         # ================ 测试首尾帧视频生成 ================
#         print("\n================== 开始测试首尾帧视频生成 ====================")
#         image_urls = [
#             "https://glrs.laihua.com/2025-10-27/韩立紫灵.jpg",
#             "https://glrs.laihua.com/2025-10-27/韩立紫灵.jpg"
#         ]
#         prompt = "女子含情脉脉的看着男子，最后忍不住亲了男子的脸颊，然后依依不舍地分开继续含情脉脉的看着男子"
#         try:
#             task_id = await generator.start_end_gen_video(image_urls, prompt)
#             print(f"✅ 任务创建成功! Task ID: {task_id}")

#             while True:
#                 status_data = await generator.check_status(task_id)
#                 current_state = status_data["state"]
#                 print(f"🔄 任务状态: {current_state} (轮询时间: {time.strftime('%Y-%m-%d %H:%M:%S')})")

#                 if current_state == "success":
#                     video_url = status_data["creations"][0]["url"]
#                     print(f"\n🎉 视频生成成功! 下载地址: {video_url}")
#                     break
#                 elif current_state == "failed":
#                     err_code = status_data.get("err_code", "未知错误")
#                     print(f"❌ 任务失败! 错误代码: {err_code}")
#                     break
#                 await asyncio.sleep(10)
#         except Exception as e:
#             print(f"⚠️ 发生异常: {repr(e)}")

        # ================ 测试参考生视频生成 ================
        print("\n================== 开始测试参考生视频 =================")
        subjects = [
            # {
            #     'id': '韩立',
            #     'images': ['https://fresource.laihua.com/2026-01-17/韩立.jpg'],
            #     'voice_id': ''
            # },
            # {
            #     'id': '紫灵',
            #     'images': ['https://fresource.laihua.com/2026-01-17/紫灵.jpg'],
            #     'voice_id': ''
            # },
            {
                'id': '场景1',
                'images': ['https://fresource.laihua.com/2026-06-01/doubao_image_1046a984bcfd42cd82fbe7a51c663d29.jpeg'],
                'voice_id': ''
            },
            {
                'id': '人物1',
                'images': ['https://fresource.laihua.com/2026-06-01/doubao_image_d78a0e0280284ffdafe0f2eca6ac81fd.jpeg'],
                'voice_id': ''
            },
        ]
#         prompt = """
# @韩立 和 @紫灵 站立在小溪旁，夜色朦胧，小溪上的莲花发出微光，@紫灵 对 @韩立 说：“韩大哥，你可愿与我结成道侣”。@韩立 深情凝视着 @紫灵 ，没有回答，只是缓慢的亲吻了 @紫灵 ，@紫灵 身体僵硬了一下，随即放松，双手搂住 @韩立 。
# """
        prompt="""
在场景 @场景1 内，@人物1 慢慢旋转，全身轮廓在阳光映照下更显柔和，叶片发饰晃动，胸前LOGO熠熠生辉，镜头由远及近，配以轻快的脚步声和微笑音效。南小慢轻声说：“‘南’取自南山。”"""

#         subjects = [
#             {
#                 "id": "nine_grid",
#                 "images": ["https://fresource.laihua.com/2026-03-27/b1068a05-07bb-4c31-9a2c-afa0b638e2a5.png"],
#                 "voice_id": ''
#             }
#         ]
#         prompt = """
# At dawn, begin with a wide establishing shot revealing a golden countryside bathed in soft morning light, where a young character with short dark hair, dressed in a simple t-shirt and shorts, stands quietly facing vast sunlit fields, the anime art style emphasizing delicate colors and intricate backgrounds. Gently push the camera forward as the character steps into the tall grass, shifting to a medium perspective where curiosity lights up their eyes; they crouch, intensely focused on a fluttering butterfly nearby, sunlight creating lively stripes and highlighting the lush field with vibrant anime textures. Transition to a low angle tracking shot from behind as the character dashes through vivid green and gold grass, arms outstretched, relentlessly chasing the butterfly—motion blur and cinematic movement immerse the viewer in this energetic pursuit. Suddenly the rush halts; cut to a close-up as the character pauses, softly panting, sweat beading from exertion, fingers brushing through grass, expressing both disappointment and renewed determination beneath sun-dappled shadows and rich anime detail.\n\nWith a subtle camera drift, shift to a medium shot as the character sits among wildflowers, knees drawn in, gazing thoughtfully toward the horizon, petals drifting in a gentle breeze—emotion and color gradients enriching the atmosphere. Over-the-shoulder, the camera captures the character’s sudden discovery of a small animal, perhaps a rabbit, peeking from the field’s edge; their posture transforms from quiet sadness to bright surprise, sharing a moment of mutual curiosity under dappled light and detailed anime brushwork. The tension builds in a dynamic side view—slowly, the character stands and carefully extends a hand toward the animal, whose uncertain yet trusting response fills the frame with gentle anticipation, grass and flowers softly framing the scene in warm cinematography.\n\nMove in for an intimate close-up as the animal cautiously approaches, sniffing the character’s outstretched palm; eyes widen with wonder and pure joy, golden sunlight highlighting expressive features, background fading into a soft bokeh—emotional resonance elevated by vivid anime colors and texture. Conclude with a wide sunset shot, the character and rabbit silhouetted against a dramatic orange sky at the edge of the field, both figures peaceful and immersed in the enchanting landscape—cinematic lighting, magical realism, and harmonious tones delivering a serene, heartwarming resolution. Throughout, maintain smooth transitions and consistent anime visual style, focusing on expressive gesture, evolving emotion, and subtle nature details, creating a cohesive, immersive short film filled with warmth and gentle discovery.
# """
        try:
            task_id = await generator.refer_gen_video(subjects, 
                                                      prompt, 
                                                      resolution="540p",
                                                      duration=8, 
                                                      model="viduq3")
            print(f"✅ 任务创建成功! Task ID: {task_id}")
            return await generator.wait_for_task_and_upload_to_obs(
                task_id,
                poll_interval=10,
                oversea=True,
                filename_prefix="vidu_reference_video",
            )
        except Exception as e:
            print(f"⚠️ 发生异常: {repr(e)}")
            return None

    result = asyncio.run(main())
    # python -m tools.video_generation.vidu.video_generation
    print(f"main() 返回的 OBS 链接: {result}")