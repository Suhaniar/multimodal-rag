import sys  #gives access to python runtime settings /modules
from pathlib import Path #path gives 
sys.path.insert(0, str(Path(__file__).parent.parent)) #gives access to root folder

import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from app.core.config import APP_TITLE
from app.core.startup import lifespan
from app.api.routes import router

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("multimodal_rag")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)
file_handler = RotatingFileHandler(
    LOG_DIR / "rag_backend.log", maxBytes=10_000_000, backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_format)
logger.addHandler(file_handler)

app = FastAPI(title=APP_TITLE, lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)