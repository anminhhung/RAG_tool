from fastapi import APIRouter, Request, HTTPException
from .service import AssistantService
import asyncio
from datetime import time

router = APIRouter()
assistant = AssistantService()

# --- API Endpoints ---
@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    return response