import yaml
import asyncio
import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from volcenginesdkarkruntime import Ark


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


def _resolve_path(path: str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return str(resolved)


class DoubaoImageToolItem(BaseModel):
    name: str = Field(..., description="主体名字")
    image_prompt: str = Field(..., min_length=1, description="主体设定图提示词")
    image: str | list[str] | None = Field(default=None, description="参考图片 URL，传入时走图生图")


class DoubaoImageToolResultItem(BaseModel):
    name: str = Field(..., description="主体名字")
    image_prompt: str = Field(..., description="主体设定图提示词")
    image_url: str | None = Field(default=None, description="生成后的图片 URL")
    image_error: str | None = Field(default=None, description="生成失败时的错误信息")
    mode: str = Field(..., description="生成模式：text_to_image 或 image_to_image")


class DoubaoImageToolBatchResult(BaseModel):
    subjects: list[DoubaoImageToolResultItem]

class DoubaoImageGenerator:
    def __init__(self, config_path="config/doubao/config.yaml"):
        """
        初始化图像生成器，从配置文件加载API Key
        
        参数:
            config_path: 配置文件路径，默认为 'config/doubao/config.yaml'
        """
        self.config_path = _resolve_path(config_path)
        self.config = self._load_config()
        self.api_key = self.config["API_KEY"]
        self.image_model = self.config["image_model"]
        self.client = self._initialize_client()

    def _load_config(self):
        """从 YAML 配置文件中加载必需配置。"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"配置文件未找到: {self.config_path}")

        if not isinstance(config, dict):
            raise Exception(f"配置文件格式无效: {self.config_path}")

        required_keys = ["API_KEY", "image_model"]
        missing_keys = [key for key in required_keys if not config.get(key)]
        if missing_keys:
            raise Exception(f"配置文件中缺少字段: {', '.join(missing_keys)}")

        return config

    def _initialize_client(self):
        """初始化方舟客户端"""
        return Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )

    def generate_image(self, 
                       prompt, 
                       image=None,
                              model=None,
                       size="2K", 
                       watermark=False
                       ):
        """
        生成图像并返回URL
        
        参数:
            prompt: 图像描述提示词
            model: 使用的模型，默认为配置文件中的 image_model
            size: 图像尺寸，默认为 '2K'
            watermark: 是否添加水印，默认为 True
            
        返回:
            生成的图像URL
        """
        model = model or self.image_model

        # 准备API调用参数
        api_params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "watermark": watermark
        }
        
        # 根据是否提供image参数决定调用方式
        if image is not None:
            # 图生图模式：添加image参数
            api_params["image"] = image
            # print("使用图生图模式")
        else:
            # 文生图模式：不添加image参数
            # print("使用文生图模式")
            pass
        
        # 调用API
        response = self.client.images.generate(**api_params)
        return response.data[0].url
    
    async def async_generate_image(self, 
                                  prompt, 
                                  image=None,
                                  model=None,
                                  size="2K", 
                                  watermark=False
                                  ):
        """
        异步生成图像并返回URL
        
        参数:
            prompt: 图像描述提示词
            image: 参考图像URL（图生图时使用）
            model: 使用的模型，默认为配置文件中的 image_model
            size: 图像尺寸，默认为 '2K'
            watermark: 是否添加水印，默认为 False
            
        返回:
            生成的图像URL
        """
        try:
            # 使用线程池执行同步的API调用，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.generate_image, 
                prompt, image, model, size, watermark
            )
            # print("工具类函数--》豆包生成图片返回的结果", result)
            return result
        except Exception as e:
            print(f"异步图像生成失败: {repr(e)}")
            raise
    
    async def async_generate_text_to_image(self, 
                                         prompt, 
                                         model=None,
                                         size="2K", 
                                         watermark=False
                                         ):
        """
        异步文生图方法
        
        参数:
            prompt: 图像描述提示词
            model: 使用的模型
            size: 图像尺寸
            watermark: 是否添加水印
            
        返回:
            生成的图像URL
        """
        # print("使用异步文生图模式")
        return await self.async_generate_image(
            prompt=prompt,
            image=None, 
            model=model,
            size=size,
            watermark=watermark
        )

    async def async_generate_image_to_image(self, 
                                          prompt, 
                                          image=None,
                                                                                    model=None,
                                          size="2K", 
                                          watermark=False
                                          ):
        """
        异步图生图方法
        
        参数:
            prompt: 图像描述提示词
            image: 参考图像URL
            model: 使用的模型
            size: 图像尺寸
            watermark: 是否添加水印
            
        返回:
            生成的图像URL
        """
        # print("使用异步图生图模式")
        return await self.async_generate_image(
            prompt=prompt,
            image=image, 
            model=model,
            size=size,
            watermark=watermark
        )


@tool(parse_docstring=True)
async def generate_doubao_images_tool(
    subjects: list[dict],
    config_path: str = "config/doubao/config.yaml",
    model: str | None = None,
    size: str = "2K",
    watermark: bool = False,
) -> str:
    """批量调用豆包生成主体设定图。

    Args:
        subjects: 主体列表，每项至少需要包含 name 和 image_prompt，可选 image 作为图生图参考图。
        config_path: 豆包配置路径。
        model: 生图模型名，为空时使用配置中的默认模型。
        size: 图片尺寸。
        watermark: 是否加水印。
    """
    generator = DoubaoImageGenerator(config_path=config_path)
    payload = [DoubaoImageToolItem.model_validate(subject) for subject in subjects]

    async def _generate(subject: DoubaoImageToolItem) -> DoubaoImageToolResultItem:
        try:
            if subject.image is None:
                image_url = await generator.async_generate_text_to_image(
                    prompt=subject.image_prompt,
                    model=model,
                    size=size,
                    watermark=watermark,
                )
                mode = "text_to_image"
            else:
                image_url = await generator.async_generate_image_to_image(
                    prompt=subject.image_prompt,
                    image=subject.image,
                    model=model,
                    size=size,
                    watermark=watermark,
                )
                mode = "image_to_image"
            return DoubaoImageToolResultItem(
                name=subject.name,
                image_prompt=subject.image_prompt,
                image_url=image_url,
                image_error=None,
                mode=mode,
            )
        except Exception as exc:
            return DoubaoImageToolResultItem(
                name=subject.name,
                image_prompt=subject.image_prompt,
                image_url=None,
                image_error=repr(exc),
                mode="image_to_image" if subject.image is not None else "text_to_image",
            )

    result = await asyncio.gather(*(_generate(subject) for subject in payload))
    return DoubaoImageToolBatchResult(subjects=list(result)).model_dump_json(ensure_ascii=False)

# 使用示例
if __name__ == "__main__":
    # 初始化图像生成器
    generator = DoubaoImageGenerator()
    
    # 设置提示词：拆分为主体外貌、三视图与配饰通用要求、图片风格三个部分，方便和模型生成的人物描述拼接
    subject_appearance_prompt = (
        """
    - 主体外貌描述：\n
        - 骑士团步兵，男性，年轻至中年成人，体格匀称结实，长期训练形成稳定耐用的军人姿态，整体气质朴素、克制、服从纪律，没有明显的主角感或传奇英雄气质。\n
        - 发型保持军旅风格的简洁短发，颜色自然，发束整齐，便于佩戴头盔与日常行军，不做夸张造型。\n
        - 面部带有普通士兵常见的风吹日晒痕迹，五官端正但不锐利，神情沉稳务实，眼神警觉而内敛，给人可靠但低调的印象。\n
        - 上身穿制式骑士团步兵轻中型甲胄，包含内衬布衣、皮革固定带、金属胸甲与肩甲，结构以实用防护为主，没有夸张装饰，仅保留少量骑士团徽记或编号细节。\n
        - 手臂佩戴基础护臂与皮质手套，服装整体以灰蓝、铁灰、暗棕和旧银色为主，材质偏向布料、皮革与磨损金属，体现长期执勤和战场使用痕迹。\n
        - 下身穿便于行军作战的长裤、护膝与结实军靴，腰部配有制式腰带、简易挂扣、小型杂物包或水袋，整体装备符合普通步兵的标准配置。\n
        - 武器为常规制式长剑、短矛或圆盾等骑士团步兵常见装备，尺寸合理，强调耐用性与量产感，不要出现夸张厚重的传奇武器。\n
        - 整体角色外观需要明确、完整，呈现为一名真实可信、身份普通的骑士团基层步兵，适合直接用于高质量角色设定图生成。
"""
    )

    three_view_accessories_prompt = (
        """
    - 画面任务：\n
        - 生成一张完整的主体设定展示图，重点展示同一主体的三视图与随身物件。\n
    - 三视图要求：\n
        - 主画面中并排展示主体正面、侧面、背面三视图。\n
        - 三个视角中的主体必须保持同一设定，例如五官、发型、服装、体型、比例、配色、材质、物件等必须完全一致，仅视角发生变化。\n
        - 主体需完整清晰，适合做主体设定展示。\n
    - 物件展示要求：\n
        - 在主体三视图周围补充展示该角色佩戴的物件与特征细节。\n
        - 物件必须完整展示出来，不能被遮挡或部分展示，必须清晰可见，适合做物件细节展示。\n
        - 禁止出现主体没有佩戴的物件；\n
        - 禁止出现主体的内衣内搭，必须展示与主体三视图中完全一致的穿戴展示；\n
        - 纹身、徽章、伤痕、布料纹样等特征细节必须在主体三视图中同样清晰可辨，并保持位置、形状、颜色、数量完全一致。\n
    - 版式与画面要求：\n
        - 整体像游戏或影视项目中的角色设定稿，排版整洁，主体明确，层次清楚。\n
        - 纯净浅色背景，无复杂场景，无多人，无文字说明，无水印。\n
        - 强调主体设计一致性、高清细节、工业级设定图质感。
"""
    )

    style_prompt = (
        """
    - 图片风格：\n
        - Japanese anime
"""
    )

    prompt = "\n".join([
        subject_appearance_prompt.strip(),
        three_view_accessories_prompt.strip(),
        style_prompt.strip(),
    ])
# Photorealistic CG Film   # CG电影
# Cultivation Fantasy 3D    # 国风3D
# Japanese anime  # 日式动漫
#     prompt = (
#         """
#     生成一个标准的九宫格3乘以3的图片，内容全部是灰色, 格子与格子之间有小箭头连接代表格子的顺序
# """
#     )
#     prompt = (
#         """
#     按照该三视图风格生成一个大叔的三视图
# """
#     )
    
    # 生成图像并获取URL
    # image = [
    #     # "https://fresource.laihua.com/2025-11-11/da4cbff838084b5590c3d521ac281b68.jpg",
    #     # "https://fresource.laihua.com/2025-11-11/ee24b370cd794ef0b434b733b7ec7a70.jpg"
    #     "https://glrs.laihua.com/2025-11-21/42ee8850179f8.jpeg"
    # ]
    async def main():
        image_url = await generator.async_generate_text_to_image(prompt)
        print("豆包返回的图像URL:", image_url)

    asyncio.run(main())