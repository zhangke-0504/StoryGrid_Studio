import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from volcenginesdkarkruntime import Ark

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(path: str) -> str:
	resolved = Path(path)
	if not resolved.is_absolute():
		resolved = PROJECT_ROOT / resolved
	return str(resolved)

from utils.llm_factory import build_chat_openai, with_llm_retry
from utils.retry import async_retry
from tools.image_generation.doubao.image_generation import DoubaoImageGenerator


class PromptAuditExtractionResult(BaseModel):
	audit_points: List[str] = Field(
		...,
		description="需要重点审核的点，尤其是文字、数字、专有名词、标识和关键视觉元素。",
	)
	audit_prompt: str = Field(
		...,
		description="提供给图片理解模型的审核提示词，要求逐项判断生成结果是否符合原始提示词。",
	)


class ReconstructionPromptResult(BaseModel):
	should_reconstruct: bool = Field(
		...,
		description="是否需要进入图生图重构流程。true 表示需要重构，false 表示原始图片可直接使用。",
	)
	reconstruction_prompt: str = Field(
		...,
		description="用于图生图重构的最终提示词，既保留正确内容，也重点修正审核中发现的问题。",
	)
	repair_focus: List[str] = Field(
		...,
		description="本次图生图重构需要重点修复的要点列表。",
	)


class ImageReconstructionWorkflowResult(BaseModel):
	audit_result: PromptAuditExtractionResult
	understanding_result: str = Field(
		...,
		description="图片理解审核结果文本。",
	)
	reconstruction_result: ReconstructionPromptResult
	origin_image_url: str = Field(
		...,
		description="原始图片链接。",
	)
	reconstructed_image_url: Optional[str] = Field(
		default=None,
		description="重构后的图片链接；如果无需重构则为空。",
	)
	elapsed_seconds: float = Field(
		...,
		description="整个流程耗时，单位秒。",
	)


class ImageAuditToolItem(BaseModel):
	name: str = Field(..., description="主体名字")
	image_prompt: str = Field(..., min_length=1, description="原始生图提示词")
	image_url: str = Field(..., min_length=1, description="原始图片链接")


class ImageAuditToolResultItem(BaseModel):
	name: str = Field(..., description="主体名字")
	image_prompt: str = Field(..., description="原始生图提示词")
	image_url: str = Field(..., description="原始图片链接")
	audit_points: List[str] = Field(default_factory=list, description="审核重点")
	understanding_result: Optional[str] = Field(default=None, description="图片理解审核结果")
	should_reconstruct: bool = Field(default=False, description="是否需要图生图重构")
	reconstruction_prompt: Optional[str] = Field(default=None, description="图生图重构提示词")
	repair_focus: List[str] = Field(default_factory=list, description="重构重点修复项")
	reconstructed_image_url: Optional[str] = Field(default=None, description="重构后的图片 URL")
	final_image_url: Optional[str] = Field(default=None, description="最终可用图片 URL")
	error: Optional[str] = Field(default=None, description="审核或重构失败时的错误信息")


class ImageAuditToolBatchResult(BaseModel):
	subjects: List[ImageAuditToolResultItem]


class OriginImagePromptAuditor:
	def __init__(self, openai_config_path: str = "config/openai/config.yaml"):
		self.openai_config_path = openai_config_path

	@async_retry(max_attempts=3)
	async def async_extract_audit_prompt(
		self,
		origin_prompt: str,
		language: str = "zh-CN",
	) -> PromptAuditExtractionResult:
		# 原始生图提示词没有明确说明物品具体名称的，审核意见要强调该物品上禁止生成文字。例如原始生图提示词中出现“胸前有医院LOGO缩影”，则审核意见必须提出提出“医院LOGO禁止出现文字内容”
		instructions = """
你是一个图片审核提示词提炼助手。你的任务是从原始生图提示词中提炼出最值得审核的内容，并产出一个可直接交给图片理解模型使用的审核提示词。

输出要求：
1. 必须返回 JSON 结构。
2. `audit_points` 只保留最需要审核的点，优先提取以下内容：
   - 图片里应出现且容易生成错误的文字、招牌、横幅、标识语。
   - 医院、学校、品牌、地名、机构名、人名等专有名词。
   - 数字、日期、编号、楼层、车牌、电话等容易出错的信息。
   - 对画面成立非常关键的主体、场景属性、颜色、布局、构图要求。
3. `audit_points` 每项都要是完整、清晰、可核查的中文短句。
4. `audit_prompt` 必须是一段可直接发送给图片理解模型的中文审核指令，要求模型：
   - 先整体理解图片内容。
   - 再逐条核查 `audit_points` 是否在图中正确体现。
   - 对文字内容要严格核对是否完全一致，不允许错字、漏字、近义替换。
   - 禁止生成原始生图提示词中无关的文字内容。
   - 对不存在、模糊不清、无法判断的点明确标记出来。
   - 最后给出整体审核结论。
5. 所有输出内容使用 {language}。
""".strip().format(language=language)

		prompt = f"请提炼下面原始生图提示词中的审核重点，并生成审核提示词：\n\n{origin_prompt.strip()}"
		llm = with_llm_retry(
			build_chat_openai(config_path=self.openai_config_path).with_structured_output(
				PromptAuditExtractionResult, method="function_calling"
			)
		)
		return await llm.ainvoke(
			[SystemMessage(content=instructions), HumanMessage(content=prompt)]
		)

	@async_retry(max_attempts=3)
	async def async_summarize_reconstruction_prompt(
		self,
		origin_prompt: str,
		audit_result_text: str,
		language: str = "zh-CN",
	) -> ReconstructionPromptResult:
		instructions = """
你是一个图生图重构提示词生成助手。你的任务是根据原始生图提示词和图片审核结果，总结出下一轮“图生图”需要使用的重构提示词。

输出要求：
1. 必须返回 JSON 结构。
2. `should_reconstruct` 用于判断是否需要图生图重构：
	- 如果原始图片已经满足审核要求，返回 `false`。
	- 如果原始图片存在明显问题，需要继续修正，返回 `true`。
3. `reconstruction_prompt` 必须直接可用于图生图：
   - 保留审核结果中已经正确生成的内容，不要无故改写。
   - 对审核不通过的部分进行强化、纠偏和约束。
   - 尤其要强化文字、招牌、机构名、地名、编号等易错信息的准确生成要求。
   - 如果审核结果指出文字错误，必须明确要求文字内容逐字准确，不允许错字、漏字、近义替换、形近字替换。
   - 要体现这是基于当前图片进行局部修正和整体优化的图生图提示词。
	- 如果 `should_reconstruct` 为 `false`，`reconstruction_prompt` 返回空字符串即可。
4. `repair_focus` 只列出本次必须重点修复的问题，使用简洁中文短句；如果不需要重构则返回空列表。
5. 输出语言使用 {language}。
""".strip().format(language=language)

		prompt = (
			"请根据下面的原始生图提示词和图片审核结果，整理出下一轮图生图重构提示词。\n\n"
			f"原始生图提示词：\n{origin_prompt.strip()}\n\n"
			f"图片审核结果：\n{audit_result_text.strip()}"
		)
		llm = with_llm_retry(
			build_chat_openai(config_path=self.openai_config_path).with_structured_output(
				ReconstructionPromptResult, method="function_calling"
			)
		)
		return await llm.ainvoke(
			[SystemMessage(content=instructions), HumanMessage(content=prompt)]
		)


class DoubaoImageUnderstanding:
	def __init__(
		self,
		config_path: str = "config/doubao/config.yaml",
		model: str = "doubao-seed-2-0-lite-260215",
	):
		self.config_path = _resolve_path(config_path)
		self.api_key = self._load_api_key()
		self.model = model
		self.client = self._initialize_client()

	def _load_api_key(self) -> str:
		try:
			with open(self.config_path, "r", encoding="utf-8") as file:
				config = yaml.safe_load(file)
				api_key = config.get("API_KEY")
				if not api_key:
					raise ValueError("配置文件中缺少 API_KEY 字段")
				return api_key
		except FileNotFoundError as exc:
			raise FileNotFoundError(f"配置文件未找到: {self.config_path}") from exc

	def _initialize_client(self) -> Ark:
		return Ark(
			base_url="https://ark.cn-beijing.volces.com/api/v3",
			api_key=self.api_key,
		)

	def understand_image(
		self,
		prompt: str,
		image_url: str,
		system_prompt: Optional[str] = None,
	):
		input_payload = []

		if system_prompt:
			input_payload.append(
				{
					"role": "system",
					"content": [
						{
							"type": "input_text",
							"text": system_prompt,
						}
					],
				}
			)

		input_payload.append(
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": prompt,
					},
					{
						"type": "input_image",
						"image_url": image_url,
					},
				],
			}
		)

		response = self.client.responses.create(
			model=self.model,
			input=input_payload,
		)
		return response

	def _extract_response_text(self, response) -> str:
		output_text = getattr(response, "output_text", None)
		if output_text:
			return output_text

		output_items = getattr(response, "output", None)
		if output_items:
			for item in output_items:
				if getattr(item, "type", None) != "message":
					continue
				content_items = getattr(item, "content", None) or []
				text_parts = []
				for content_item in content_items:
					text_value = getattr(content_item, "text", None)
					if text_value:
						text_parts.append(text_value)
				if text_parts:
					return "\n".join(text_parts)

		return str(response)

	def understand_image_text(
		self,
		prompt: str,
		image_url: str,
		system_prompt: Optional[str] = None,
	) -> str:
		response = self.understand_image(
			prompt=prompt,
			image_url=image_url,
			system_prompt=system_prompt,
		)
		return self._extract_response_text(response)

	async def async_understand_image_text(
		self,
		prompt: str,
		image_url: str,
		system_prompt: Optional[str] = None,
	) -> str:
		loop = asyncio.get_running_loop()
		return await loop.run_in_executor(
			None,
			self.understand_image_text,
			prompt,
			image_url,
			system_prompt,
		)

	async def async_run_reconstruction_workflow(
		self,
		origin_prompt: str,
		origin_image_url: str,
		language: str = "zh-CN",
		openai_config_path: str = "config/openai/config.yaml",
		image_config_path: str = "config/doubao/config.yaml",
		reconstruction_model: Optional[str] = None,
		reconstruction_size: str = "2K",
		reconstruction_watermark: bool = False,
		system_prompt: str = "你是一个严谨的图片审核助手，需要严格核对图片与审核要求是否一致。",
	) -> ImageReconstructionWorkflowResult:
		start_time = time.perf_counter()
		prompt_auditor = OriginImagePromptAuditor(openai_config_path=openai_config_path)
		doubao_generator = DoubaoImageGenerator(config_path=image_config_path)

		audit_result = await prompt_auditor.async_extract_audit_prompt(
			origin_prompt=origin_prompt,
			language=language,
		)
		# print("审核提示词:", audit_result.audit_prompt)

		understanding_result = await self.async_understand_image_text(
			prompt=audit_result.audit_prompt,
			image_url=origin_image_url,
			system_prompt=system_prompt,
		)
		# print("图片理解审核结果:", understanding_result)

		reconstruction_result = await prompt_auditor.async_summarize_reconstruction_prompt(
			origin_prompt=origin_prompt,
			audit_result_text=understanding_result,
			language=language,
		)
		# print("图生图重构提示词:", reconstruction_result.reconstruction_prompt)

		reconstructed_image_url = None
		if reconstruction_result.should_reconstruct:
			reconstructed_image_url = await doubao_generator.async_generate_image_to_image(
				prompt=reconstruction_result.reconstruction_prompt,
				image=origin_image_url,
				model=reconstruction_model,
				size=reconstruction_size,
				watermark=reconstruction_watermark,
			)

		elapsed_seconds = time.perf_counter() - start_time
		return ImageReconstructionWorkflowResult(
			audit_result=audit_result,
			understanding_result=understanding_result,
			reconstruction_result=reconstruction_result,
			origin_image_url=origin_image_url,
			reconstructed_image_url=reconstructed_image_url,
			elapsed_seconds=elapsed_seconds,
		)


@tool(parse_docstring=True)
async def audit_and_reconstruct_images_tool(
	subjects: List[Dict[str, Any]],
	language: str = "zh-CN",
	openai_config_path: str = "config/openai/config.yaml",
	image_config_path: str = "config/doubao/config.yaml",
	reconstruction_model: Optional[str] = None,
	reconstruction_size: str = "2K",
	reconstruction_watermark: bool = False,
	system_prompt: str = "你是一个严谨的图片审核助手，需要严格核对图片与审核要求是否一致。",
) -> str:
	"""批量审核主体图片，并在需要时执行图生图重构。

	Args:
		subjects: 主体列表，每项至少需要包含 name、image_prompt、image_url。
		language: 输出语言。
		openai_config_path: OpenAI 配置路径。
		image_config_path: 豆包配置路径。
		reconstruction_model: 图生图模型名，为空时使用默认模型。
		reconstruction_size: 图生图尺寸。
		reconstruction_watermark: 图生图是否加水印。
		system_prompt: 图片理解审核系统提示词。
	"""
	understanding = DoubaoImageUnderstanding(config_path=image_config_path)
	payload = [ImageAuditToolItem.model_validate(subject) for subject in subjects]

	async def _audit(subject: ImageAuditToolItem) -> ImageAuditToolResultItem:
		try:
			workflow_result = await understanding.async_run_reconstruction_workflow(
				origin_prompt=subject.image_prompt,
				origin_image_url=subject.image_url,
				language=language,
				openai_config_path=openai_config_path,
				image_config_path=image_config_path,
				reconstruction_model=reconstruction_model,
				reconstruction_size=reconstruction_size,
				reconstruction_watermark=reconstruction_watermark,
				system_prompt=system_prompt,
			)
			return ImageAuditToolResultItem(
				name=subject.name,
				image_prompt=subject.image_prompt,
				image_url=subject.image_url,
				audit_points=workflow_result.audit_result.audit_points,
				understanding_result=workflow_result.understanding_result,
				should_reconstruct=workflow_result.reconstruction_result.should_reconstruct,
				reconstruction_prompt=workflow_result.reconstruction_result.reconstruction_prompt,
				repair_focus=workflow_result.reconstruction_result.repair_focus,
				reconstructed_image_url=workflow_result.reconstructed_image_url,
				final_image_url=workflow_result.reconstructed_image_url or workflow_result.origin_image_url,
				error=None,
			)
		except Exception as exc:
			return ImageAuditToolResultItem(
				name=subject.name,
				image_prompt=subject.image_prompt,
				image_url=subject.image_url,
				audit_points=[],
				understanding_result=None,
				should_reconstruct=False,
				reconstruction_prompt=None,
				repair_focus=[],
				reconstructed_image_url=None,
				final_image_url=subject.image_url,
				error=repr(exc),
			)

	result = await asyncio.gather(*(_audit(subject) for subject in payload))
	return ImageAuditToolBatchResult(subjects=list(result)).model_dump_json(ensure_ascii=False)



origin_image_prompt = """
设定：
无人的场景
外观：
现代风格的医院建筑全景，布局有序，主色调为白色和淡绿色，外观简洁明亮，入口显眼，门前有开放的广场与少量绿植，整体环境显得专业且温馨，能够代表深圳市南山区慢性病防治院的形象。
"""
origin_image_url = "https://fresource.laihua.com/2026-05-25/92cddec5b02e48b39a85002fae32377d.jpg"

origin_image_prompt2 = """
设定：
深圳市南山区慢性病防治院吉祥物，温暖亲切，富有耐心与爱心，使命是陪伴、守护居民健康，积极参与健康宣教、医学普及、慢病管理等公益活动，是健康理念的传递者和朋友。
外观：
卡通角色，整体圆润可爱，主色调为健康绿与温暖橙，头顶佩戴叶片发饰，胸前有医院LOGO缩影，活泼微笑，五官柔和，手臂常做挥手、比心、加油等友善动作。
"""
origin_image_url2 = "https://fresource.laihua.com/2026-05-19/42b0ff21115640bbbe1c8ca48e8a7060.jpg"

origin_image_prompt3 = """
设定：
无人的场景
外观：
明亮宽敞的现代化医院内部场景，装修简洁，色调以浅色为主。透过大面积玻璃窗采光良好，大厅内有导诊台，墙面贴有健康科普宣传海报。门诊楼外观线条流畅，门前设有医院LOGO，周边绿植点缀，整体风格温馨而具科技感。不同科室区域有明显区分标志，如心脑血管科、精神卫生科、口腔防治科等，人与医疗环境交融，充满关怀氛围。
"""
origin_image_url3 = "https://fresource.laihua.com/2026-05-25/multi_model_16e46fc078464e69bcc0de5931117230.jpg"

origin_image_prompt4 = """
设定：
深圳市南山区慢性病防治院的医护代表，专业、负责，善于为慢病患者及社区居民提供全周期健康服务，协助南小慢共同守护公众健康。
外观：
穿着白色或浅蓝色工作服，有护士帽或医生听诊器，形象温和可靠，经常与南小慢互动如并肩工作、参与社区筛查，部分以剪影形式出现。
"""

origin_image_url4 = "https://fresource.laihua.com/2026-05-19/f3bbb7b20b7749eba5c19cb1155cdca2.jpg"

if __name__ == "__main__":

	async def main():
		doubao_understanding = DoubaoImageUnderstanding()
		workflow_result = await doubao_understanding.async_run_reconstruction_workflow(
			origin_prompt=origin_image_prompt4,
			origin_image_url=origin_image_url4,
		)

		print("提炼出的审核点:")
		for index, audit_point in enumerate(workflow_result.audit_result.audit_points, start=1):
			print(f"{index}. {audit_point}")

		print("\n审核提示词:")
		print(workflow_result.audit_result.audit_prompt)

		print("\n图片理解审核结果:")
		print(workflow_result.understanding_result)

		print("\n图生图修复重点:")
		for index, focus in enumerate(workflow_result.reconstruction_result.repair_focus, start=1):
			print(f"{index}. {focus}")

		print("\n重构前的原始图片URL:")
		print(workflow_result.origin_image_url)
		
		print("是否需要重构:", workflow_result.reconstruction_result.should_reconstruct)
		if workflow_result.reconstruction_result.should_reconstruct:
			print("\n图生图重构提示词:")
			print(workflow_result.reconstruction_result.reconstruction_prompt)
			print("\n重构后的图片URL:")
			print(workflow_result.reconstructed_image_url)
		else:
			print("\n无需图生图重构，原始图片可直接使用。")

		print(f"\n整个流程耗时: {workflow_result.elapsed_seconds:.2f} 秒")

	asyncio.run(main())

