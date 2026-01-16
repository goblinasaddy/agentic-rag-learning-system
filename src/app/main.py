from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.core.config import settings
from src.app.api import ingestion, search, chat

from contextlib import asynccontextmanager
import asyncio
from src.app.worker import background_ingestion_task

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start worker
    worker_task = asyncio.create_task(background_ingestion_task())
    yield
    # Shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Include Routers
app.include_router(ingestion.router, prefix=f"{settings.API_V1_STR}/ingest", tags=["ingestion"])
app.include_router(search.router, prefix=f"{settings.API_V1_STR}/search", tags=["search"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "project": settings.PROJECT_NAME}

@app.get("/")
async def root():
    return {"message": "Welcome to the Agentic Document Intelligence Platform"}
