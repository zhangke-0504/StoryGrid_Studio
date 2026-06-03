from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger("working_flow_router")


@router.get("/health")
async def health() -> dict:
	return {
		"status": "ok",
		"service": "working_flow",
	}



