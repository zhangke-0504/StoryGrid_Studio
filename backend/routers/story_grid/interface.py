import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from agents.shared.event_emitter import reset_event_emitter, set_event_emitter
from agents.supervisor.agents import (
	StoryVideoGenerationInput,
	StoryVideoGenerationResult,
	StoryVideoSupervisorAgent,
	_extract_json_object,
	_message_text,
	_parse_tool_message_content,
)

router = APIRouter()
logger = logging.getLogger("working_flow_router")


def _sse_frame(payload: dict[str, Any], event_name: str | None = None) -> str:
	parts: list[str] = []
	if event_name:
		parts.append(f"event: {event_name}")
	parts.append(f"data: {json.dumps(payload, ensure_ascii=False)}")
	return "\n".join(parts) + "\n\n"


def _yield_error_frame(message: str) -> AsyncIterator[str]:
	async def _single() -> AsyncIterator[str]:
		yield _sse_frame({"event": "error", "payload": message}, "error")

	return _single()


def _guess_output_type(payload: Any) -> str:
	if isinstance(payload, dict):
		if payload.get("videos"):
			return "video_group"
		if payload.get("characters"):
			return "character_group"
		if payload.get("subjects"):
			for subject in payload.get("subjects", []):
				if isinstance(subject, dict) and (
					subject.get("final_image_url")
					or subject.get("reconstructed_image_url")
					or subject.get("image_url")
				):
					return "image_group"
		if payload.get("shots"):
			for shot in payload.get("shots", []):
				if isinstance(shot, dict) and shot.get("video_url"):
					return "video_group"
			return "shot_group"
		if payload.get("video_url"):
			return "video"
		if payload.get("final_image_url") or payload.get("reconstructed_image_url") or payload.get("image_url"):
			return "image"
	if isinstance(payload, list):
		if any(isinstance(item, dict) and item.get("video_url") for item in payload):
			return "video_group"
		if any(
			isinstance(item, dict)
			and (item.get("final_image_url") or item.get("reconstructed_image_url") or item.get("image_url"))
			for item in payload
		):
			return "image_group"
	return "text"


def _extract_artifacts(payload: Any) -> list[dict[str, Any]]:
	artifacts: list[dict[str, Any]] = []
	if not isinstance(payload, dict):
		return artifacts

	for character in payload.get("characters", []) or []:
		if not isinstance(character, dict):
			continue
		image_url = (
			character.get("final_image_url")
			or character.get("reconstructed_image_url")
			or character.get("image_url")
		)
		if image_url:
			artifacts.append(
				{
					"kind": "image",
					"title": character.get("name", "未命名主体"),
					"url": image_url,
					"meta": {"uid": character.get("uid"), "type": character.get("type")},
				}
			)

	for subject in payload.get("subjects", []) or []:
		if not isinstance(subject, dict):
			continue
		image_url = (
			subject.get("final_image_url")
			or subject.get("reconstructed_image_url")
			or subject.get("image_url")
		)
		if image_url:
			artifacts.append(
				{
					"kind": "image",
					"title": subject.get("name", "未命名主体"),
					"url": image_url,
					"meta": {"image_error": subject.get("image_error")},
				}
			)

	for video in payload.get("videos", []) or []:
		if not isinstance(video, dict):
			continue
		if video.get("video_url"):
			artifacts.append(
				{
					"kind": "video",
					"title": video.get("theme", f"分镜 {video.get('sort', '-') }"),
					"url": video.get("video_url"),
					"meta": {"sort": video.get("sort"), "status": video.get("status")},
				}
			)

	for shot in payload.get("shots", []) or []:
		if not isinstance(shot, dict):
			continue
		if shot.get("video_url"):
			artifacts.append(
				{
					"kind": "video",
					"title": shot.get("theme", f"分镜 {shot.get('sort', '-') }"),
					"url": shot.get("video_url"),
					"meta": {"sort": shot.get("sort"), "status": shot.get("status")},
				}
			)

	return artifacts


@router.post("/sse")
async def story_grid_stream(request: Request):
	"""流式返回 supervisor、子 agent、工具调用与最终结果。"""
	try:
		user_payload = await request.json()
		try:
			parsed_input = StoryVideoGenerationInput.model_validate(user_payload)
		except ValidationError as ve:
			return StreamingResponse(_yield_error_frame(str(ve)), media_type="text/event-stream")

		supervisor = StoryVideoSupervisorAgent()

		async def event_stream_gen():
			event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
			step_counter = {"value": 0}

			def next_step() -> int:
				step_counter["value"] += 1
				return step_counter["value"]

			async def push(event: dict[str, Any]) -> None:
				await event_queue.put(event)

			async def subagent_emitter(event: dict[str, Any]) -> None:
				event = dict(event)
				event.setdefault("step", next_step())
				event.setdefault("output_type", _guess_output_type(event.get("payload")))
				event["artifacts"] = _extract_artifacts(event.get("payload"))
				await push(event)

			token = set_event_emitter(subagent_emitter)
			final_text_holder = {"value": ""}
			character_result_holder: dict[str, Any] = {"value": None}
			shot_result_holder: dict[str, Any] = {"value": None}

			async def producer() -> None:
				try:
					await push(
						{
							"event": "agent_start",
							"name": "supervisor",
							"step": next_step(),
							"payload": {"request": parsed_input.model_dump(mode="json")},
							"output_type": "text",
						}
					)
					seen_tool_messages: set[str] = set()
					seen_tool_call_ids: set[str] = set()

					async for chunk in supervisor.agent.astream(
						{
							"messages": [
								{"role": "user", "content": supervisor._build_react_user_prompt(parsed_input)}
							]
						},
						stream_mode="values",
					):
						latest_message = chunk["messages"][-1]
						tool_calls = getattr(latest_message, "tool_calls", None) or []
						message_text = _message_text(latest_message)

						for tc in tool_calls:
							tc_id = tc.get("id") or f"{tc.get('name')}-{len(seen_tool_call_ids)}"
							if tc_id in seen_tool_call_ids:
								continue
							seen_tool_call_ids.add(tc_id)
							await push(
								{
									"event": "tool_call",
									"name": tc.get("name", "unknown_tool"),
									"step": next_step(),
									"payload": {
										"args": tc.get("args", {}),
										"id": tc.get("id"),
										"type": tc.get("type"),
									},
									"output_type": "text",
								}
							)

						if latest_message.__class__.__name__ == "ToolMessage":
							tool_message_id = getattr(latest_message, "id", None) or getattr(
								latest_message, "tool_call_id", None
							)
							if tool_message_id in seen_tool_messages:
								continue
							seen_tool_messages.add(tool_message_id)

							tool_name = getattr(latest_message, "name", "unknown_tool")
							payload = _parse_tool_message_content(getattr(latest_message, "content", ""))
							if tool_name == "generate_story_characters_subagent" and isinstance(payload, dict):
								character_result_holder["value"] = payload
							elif tool_name == "generate_story_videos_subagent" and isinstance(payload, dict):
								shot_result_holder["value"] = payload
							await push(
								{
									"event": "tool_result",
									"name": tool_name,
									"step": next_step(),
									"payload": payload,
									"output_type": _guess_output_type(payload),
									"artifacts": _extract_artifacts(payload),
								}
							)
							continue

						if message_text:
							await push(
								{
									"event": "agent_call",
									"name": "supervisor",
									"step": next_step(),
									"payload": message_text,
									"output_type": "text",
								}
							)
							final_text_holder["value"] = message_text

					final_text = final_text_holder["value"]
					final_payload: Any = None
					if final_text:
						try:
							candidate = StoryVideoGenerationResult.model_validate(
								_extract_json_object(final_text)
							).model_dump(mode="json")
							if candidate.get("videos") or candidate.get("shots"):
								final_payload = candidate
						except Exception:
							pass

					character_payload = character_result_holder["value"]
					shot_payload = shot_result_holder["value"]
					if character_payload and shot_payload:
						merged = {
							"characters": character_payload.get("characters", []),
							"plan": shot_payload.get("plan"),
							"shots": shot_payload.get("shots", []),
							"videos": shot_payload.get("videos", []),
						}
						if final_payload is None or not final_payload.get("videos"):
							final_payload = merged
						elif isinstance(final_payload, dict) and merged.get("videos"):
							# 确保最终结果中包含真实的视频结果，而不是 supervisor LLM 可能遗漏或伪造的版本
							final_payload["videos"] = merged["videos"]

					if final_payload is None and final_text:
						try:
							final_payload = json.loads(final_text)
						except Exception:
							final_payload = final_text

					if final_payload is not None:
						await push(
							{
								"event": "final_result",
								"name": "supervisor",
								"step": next_step(),
								"payload": final_payload,
								"output_type": "final",
							}
						)

					await push(
						{
							"event": "done",
							"name": "supervisor",
							"step": next_step(),
							"payload": {"status": "completed"},
							"output_type": "text",
						}
					)
				except Exception as exc:  # noqa: BLE001
					logger.exception("SSE producer error:")
					await push({"event": "error", "payload": str(exc)})
				finally:
					await event_queue.put(None)

			producer_task = asyncio.create_task(producer())
			try:
				while True:
					event = await event_queue.get()
					if event is None:
						break
					event_name = event.get("event", "message")
					yield _sse_frame(event, event_name)
			finally:
				reset_event_emitter(token)
				if not producer_task.done():
					producer_task.cancel()
				try:
					await producer_task
				except (asyncio.CancelledError, Exception):
					pass

		return StreamingResponse(
			event_stream_gen(),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no",
			},
		)
	except Exception as e:
		logger.exception("story_grid_stream error:")
		return StreamingResponse(_yield_error_frame(str(e)), media_type="text/event-stream")


@router.get("/health")
async def health() -> dict:
	return {
		"status": "ok",
		"service": "working_flow",
	}
