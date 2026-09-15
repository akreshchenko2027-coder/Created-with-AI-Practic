from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"

class Turn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []

def load_context() -> str:
    return CONTEXT_FILE.read_text(encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")

@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Повідомлення не може бути порожнім.")
        
    history = [turn.model_dump() for turn in payload.history]
    
    try:
        return llm.ask(payload.message, history, load_context())
    except llm.LLMError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Внутрішня помилка сервера.")