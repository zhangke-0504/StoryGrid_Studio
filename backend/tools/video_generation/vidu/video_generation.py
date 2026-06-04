import os
import time
import yaml
import asyncio
import aiohttp  # 用aiohttp替换httpx
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    markers = {"pyproject.toml", ".git"}
    while True:
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:
            return Path.cwd()
        current = current.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(path: str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return str(resolved)


class ViduToolCharacter(BaseModel):
    uid: str = Field(..., description="主体 UID")
    name: str = Field(..., description="主体名字")
    voice_id: str = Field(default="", description="音色 ID")
    image_url: str | None = Field(default=None, description="原始图片 URL")
    reconstructed_image_url: str | None = Field(default=None, description="重构图片 URL")
    final_image_url: str | None = Field(default=None, description="最终图片 URL")


class ViduToolShot(BaseModel):
    sort: int = Field(..., description="分镜序号")
    duration: int = Field(..., description="分镜时长")
    theme: str = Field(..., description="分镜主题")
    description: str = Field(..., description="分镜视频提示词")


class ViduVideoToolResultItem(BaseModel):
    sort: int = Field(..., description="分镜序号")
    theme: str = Field(..., description="分镜主题")
    prompt: str = Field(..., description="提交给 Vidu 的最终提示词")
    subjects: List[Dict[str, Any]] = Field(default_factory=list, description="提交给 Vidu 的主体列表")
    task_id: str | None = Field(default=None, description="Vidu 任务 ID")
    status: str = Field(..., description="submitted、success、failed")
    video_url: str | None = Field(default=None, description="成功时的视频 URL")
    error: str | None = Field(default=None, description="失败时的错误信息")


class ViduVideoToolBatchResult(BaseModel):
    shots: List[ViduVideoToolResultItem]


UID_REFERENCE_PATTERN = re.compile(r"@\^([^\^]+)\^")


def _extract_referenced_uids(description: str) -> List[str]:
    return list(dict.fromkeys(UID_REFERENCE_PATTERN.findall(description)))


def _resolve_character_image(character: ViduToolCharacter) -> str | None:
    return character.final_image_url or character.reconstructed_image_url or character.image_url


def _build_vidu_prompt(description: str) -> str:
    return UID_REFERENCE_PATTERN.sub(lambda match: f"@{match.group(1)}", description)


def _build_vidu_subjects(shot: ViduToolShot, characters: List[ViduToolCharacter]) -> List[Dict[str, Any]]:
    referenced_uids = set(_extract_referenced_uids(shot.description))
    subjects: List[Dict[str, Any]] = []
    for character in characters:
        if character.uid not in referenced_uids:
            continue
        image_url = _resolve_character_image(character)
        if not image_url:
            continue
        subjects.append(
            {
                "id": character.uid,
                "images": [image_url],
                "voice_id": character.voice_id or "",
            }
        )
    return subjects

class ViduVideoGenerator:
    def __init__(self, config_path="config/vidu/config.yaml"):
        """初始化，自动从 YAML 配置文件加载 API_KEY"""
        self.config_info = self._load_config(_resolve_path(config_path))
        # self.model = self.config_info['MODEL']
        self.model = "viduq2"
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

    async def wait_for_task_completion(
        self,
        task_id: str,
        poll_interval: int = 10,
    ) -> Optional[str]:
        """轮询视频任务，成功后返回 Vidu 提供的视频链接。"""
        while True:
            status_data = await self.check_status(task_id)
            current_state = status_data["state"]
            print(f"🔄 任务状态: {current_state} (轮询时间: {time.strftime('%Y-%m-%d %H:%M:%S')})")

            if current_state == "success":
                video_url = status_data["creations"][0]["url"]
                print(f"Vidu 视频链接: {video_url}")
                return video_url

            if current_state == "failed":
                print("生成失败的时候，返回的结构是：", status_data)
                err_code = status_data.get("err_code", "未知错误")
                print(f"❌ 任务失败! 错误代码: {err_code}")
                return None

            await asyncio.sleep(poll_interval)


@tool(parse_docstring=True)
async def generate_vidu_videos_tool(
    shots: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
    config_path: str = "config/vidu/config.yaml",
    resolution: str = "540p",
    model: str = "viduq3",
    audio: bool = True,
    aspect_ratio: str = "16:9",
    wait_for_completion: bool = True,
    poll_interval: int = 10,
) -> str:
    """为多个分镜批量提交 Vidu 视频生成任务。

    Args:
        shots: 分镜列表，每项至少需要包含 sort、duration、theme、description。
        characters: 主体列表，需要包含 uid 以及可用图片 URL 字段。
        config_path: Vidu 配置路径。
        resolution: 视频分辨率。
        model: Vidu 模型名。
        audio: 是否生成音频。
        aspect_ratio: 视频比例。
        wait_for_completion: 是否轮询等待视频完成。
        poll_interval: 轮询间隔秒数。
    """
    generator = ViduVideoGenerator(config_path=config_path)
    parsed_shots = [ViduToolShot.model_validate(shot) for shot in shots]
    parsed_characters = [ViduToolCharacter.model_validate(character) for character in characters]

    results: List[ViduVideoToolResultItem] = []
    for shot in parsed_shots:
        subjects = _build_vidu_subjects(shot, parsed_characters)
        prompt = _build_vidu_prompt(shot.description)
        if not subjects:
            results.append(
                ViduVideoToolResultItem(
                    sort=shot.sort,
                    theme=shot.theme,
                    prompt=prompt,
                    subjects=[],
                    task_id=None,
                    status="failed",
                    video_url=None,
                    error="该分镜缺少可用的主体图片，无法提交 Vidu 视频生成。",
                )
            )
            continue

        try:
            task_id = await generator.refer_gen_video(
                subjects=subjects,
                prompt=prompt,
                resolution=resolution,
                duration=shot.duration,
                audio=audio,
                aspect_ratio=aspect_ratio,
                model=model,
            )
            if wait_for_completion:
                video_url = await generator.wait_for_task_completion(task_id, poll_interval=poll_interval)
                status = "success" if video_url else "failed"
            else:
                video_url = None
                status = "submitted"

            results.append(
                ViduVideoToolResultItem(
                    sort=shot.sort,
                    theme=shot.theme,
                    prompt=prompt,
                    subjects=subjects,
                    task_id=task_id,
                    status=status,
                    video_url=video_url,
                    error=None if status != "failed" else "Vidu 任务执行失败。",
                )
            )
        except Exception as exc:
            results.append(
                ViduVideoToolResultItem(
                    sort=shot.sort,
                    theme=shot.theme,
                    prompt=prompt,
                    subjects=subjects,
                    task_id=None,
                    status="failed",
                    video_url=None,
                    error=repr(exc),
                )
            )

    return ViduVideoToolBatchResult(shots=results).model_dump_json(ensure_ascii=False)


if __name__ == "__main__":
    async def main():
        generator = ViduVideoGenerator()
        video_url = None

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
            video_url = await generator.wait_for_task_completion(task_id, poll_interval=10)
            return video_url
        except Exception as e:
            print(f"⚠️ 发生异常: {repr(e)}")
            return None

    result = asyncio.run(main())
    # python -m tools.video_generation.vidu.video_generation
    print(f"main() 返回的 Vidu 链接: {result}")