from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title='jinx_—_autonomous_engineering_agent', version='1.0.0')

from .routers.emb import router as emb_router
from .routers.log import router as log_router

app.include_router(emb_router, prefix='/emb', tags=['emb'])
app.include_router(log_router, prefix='/log', tags=['log'])
