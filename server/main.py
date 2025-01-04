import logging
import os
import asyncio
import uvicorn
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.models import ModelsDispatcher
from app.api_router import router as api_v1_models_router
from app.schemas import StatusResponse

os.environ['TORCH_HOME'] = '/data/torch_cache'
os.environ['TFHUB_CACHE_DIR'] = '/data/tfhub_cache'
os.environ['TRANSFORMERS_CACHE'] = '/data/transformers_cache'

os.makedirs('/logs', exist_ok=True)

# Configure log file path and rotating file handler
log_file = "/logs/server.log"
handler = RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10 MB per file, 5 backups
)
# Creating the logging format
log_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)
handler.setFormatter(log_formatter)
# Setting up the root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler, logging.StreamHandler()]
)

logger = logging.getLogger("main")

models_dispatcher = ModelsDispatcher()
app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

# Middleware for propper request/response logging
class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url} {request.body()}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code} for {request.method} {request.url}")
        return response
app.add_middleware(LogRequestsMiddleware)


# We load pretrained models on the server start
@app.on_event("startup")
async def startup_event():
    logger.info("Server is starting...")
    models_dispatcher.load_models()

# Saving the trained models for the next server start
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shut down")
    models_dispatcher.save_all_models()


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(status="App is on and running!")


app.include_router(api_v1_models_router, prefix="/api/v1/models")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
