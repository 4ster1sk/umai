import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.data import load_data
from app.llm import DEFAULT_MODELS, PROVIDER, stream_analyze

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/data")
def api_data() -> dict:
    data = load_data()
    model = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))
    return {
        "provider": PROVIDER,
        "model": model,
        "categories": {
            category: [entry["item"] for entry in items]
            for category, items in data.items()
        },
    }


@app.post("/api/analyze")
def api_analyze() -> StreamingResponse:
    data = load_data()
    return StreamingResponse(stream_analyze(data), media_type="text/plain; charset=utf-8")
