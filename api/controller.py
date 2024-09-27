from fastapi import APIRouter, Request
from .service import ContextualRagAgent
from src.tools.booking import load_booking_tools

router = APIRouter()
assistant = ContextualRagAgent()

# You can also add more tools for agent to use.
# This is an example of how you can add tools to the agent and guide you how the agent will route between them.
assistant.add_tools(load_booking_tools())


# --- API Endpoints ---
@router.post("/complete")
async def complete_text(request: Request):
    data = await request.json()
    message = data.get("message")
    response = assistant.predict(message)
    return response
