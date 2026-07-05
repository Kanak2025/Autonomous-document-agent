import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.config import settings
from app.models import AgentRequest, AgentResponse
from app.planner import run_agent_pipeline, GuardrailError
from app.doc_generator import build_document
from app.llm_client import get_rate_limit_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agent.api")

app = FastAPI(
    title="Autonomous Document Agent",
    description="Accepts a natural-language request, plans, executes, self-checks, "
                "and returns a generated .docx business document.",
    version="1.0.0",
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(settings.GROQ_API_KEY),
        "model": settings.GROQ_MODEL,
        "rate_limit": get_rate_limit_info(),
    }


@app.post("/agent", response_model=AgentResponse)
def run_agent(payload: AgentRequest):
    request_text = payload.request
    logger.info(f"Received request: {request_text[:120]!r}")

    try:
        plan, section_content, reflection = run_agent_pipeline(request_text)
    except GuardrailError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - top-level safety net for a demo API
        logger.exception("Unhandled agent failure")
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {e}")

    filename = build_document(plan, section_content, request_text)

    word_counts = {s: len(section_content.get(s, "").split()) for s in plan.sections}

    message = (
        f"Generated a {plan.document_type.replace('_', ' ')} titled '{plan.title}' "
        f"with {len(plan.sections)} sections."
    )
    if reflection.issues:
        message += f" Self-check flagged {len(reflection.issues)} issue(s); "
        message += "repaired before returning." if reflection.revised_sections else "returned as informational notes."

    return AgentResponse(
        status="success",
        message=message,
        plan=plan,
        reflection=reflection,
        document_filename=filename,
        download_url=f"/files/{filename}",
        section_word_counts=word_counts,
        engine_meta={"model": settings.GROQ_MODEL, "provider": "groq"},
        rate_limit=get_rate_limit_info(),
    )


@app.get("/files/{filename}")
def download_file(filename: str):
    filepath = os.path.join(settings.OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )