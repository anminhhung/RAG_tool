from fastapi import APIRouter, Request
from .service import ContextualRagReactAgent

router = APIRouter()
assistant = ContextualRagReactAgent()


# --- API Endpoints ---
@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    return response
