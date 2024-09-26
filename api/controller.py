from fastapi import APIRouter, Request
from .service import ContextualRagAgent

router = APIRouter()
assistant = ContextualRagAgent()


# --- API Endpoints ---
@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    return response
