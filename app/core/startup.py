from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import CrossEncoder
from .config import RERANKER_MODEL, APP_TITLE
import app.core.state as state
import logging

logger = logging.getLogger("multimodal_rag")

@asynccontextmanager
async def lifespan(app: FastAPI):

    state.load_state()

    logger.info(f"Loading cross encoder {RERANKER_MODEL}")

    state.reranker = CrossEncoder(RERANKER_MODEL)

    logger.info("Cross encoder loaded.")

    yield

    state.save_state()

    logger.info("State saved.")