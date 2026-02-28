"""
WarpSchema — Self-Healing Data Proxy
Copyright (c) 2026 Tarek Clarke. All Rights Reserved.

FastAPI service exposing a /normalize endpoint that auto-remediates
schema drift in real-time using BERT embeddings.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import DEFAULT_SCHEMA, TARGET_SCHEMAS, SchemaGuardConfig
from engine import TransformationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("warpschema.app")

# ---------------------------------------------------------------------------
# Application lifespan — load the model once at startup
# ---------------------------------------------------------------------------
engine: TransformationEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initialising WarpSchema engine …")
    config = SchemaGuardConfig()
    engine = TransformationEngine(config)
    logger.info("WarpSchema engine ready.")
    yield
    logger.info("Shutting down WarpSchema engine.")


app = FastAPI(
    title="WarpSchema",
    version="1.0.0",
    description="Self-healing data proxy — auto-remediates schema drift with BERT.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class NormalizeRequest(BaseModel):
    payload: dict[str, Any] = Field(
        ..., description="The raw JSON payload with potentially drifted keys."
    )
    schema_name: str = Field(
        default=DEFAULT_SCHEMA,
        description="Target schema to normalise against.",
    )


class MappingDetail(BaseModel):
    source: str
    target: str | None
    similarity: float
    exact_match: bool


class NormalizeResponse(BaseModel):
    data: dict[str, Any]
    mappings: list[MappingDetail]
    unmapped_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/normalize", response_model=NormalizeResponse)
async def normalize(request: NormalizeRequest) -> NormalizeResponse:
    """Accept a raw payload and return the schema-remediated version."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")

    try:
        result = engine.normalize(
            payload=request.payload,
            schema_name=request.schema_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return NormalizeResponse(**result.to_dict())


@app.get("/schemas")
async def list_schemas() -> dict[str, Any]:
    """Return all available target schemas."""
    return {"schemas": TARGET_SCHEMAS}


@app.get("/health")
async def health() -> dict[str, str]:
    status = "ok" if engine is not None else "not_ready"
    return {"status": status}
