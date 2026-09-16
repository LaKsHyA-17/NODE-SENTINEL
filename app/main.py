import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes_ingest import router as ingest_router
from app.api.routes_graph import router as graph_router
from app.api.routes_analytics import router as analytics_router
from app.api.routes_alerts import router as alerts_router
from app.api.routes_search import router as search_router
from app.api.routes_cdr import router as cdr_router
from app.api.routes_financial import router as financial_router
from app.api.routes_timeline import router as timeline_router
from app.api.routes_risk import router as risk_router
from app.api.routes_assistant import router as assistant_router
from app.api.routes_reports import router as reports_router
from app.api.routes_auth import router as auth_router
from app.api.routes_users import router as users_router
from app.api.routes_audit import router as audit_router
from app.api.routes_cross_domain import router as cross_domain_router
from app.api.routes_evidence import router as evidence_router
from app.core.audit_logger import audit_logger
from app.models.audit_models import AuditAction
from app.core.graph_engine import get_graph_engine
from app.core.face_storage import get_face_storage
from app.core.demo_face_data import seed_demo_face_database
from app.api.routes_ingest import ingest_sample_batch_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from app.api.routes_ingest import load_dataset_by_name
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Auto-load synthetic dataset on server startup if graph engine is empty and record startup audit."""
    audit_logger.log(
        action=AuditAction.SYSTEM_STARTUP,
        status="SUCCESS",
        details={"version": "1.0.0", "project": settings.PROJECT_NAME},
    )
    graph = get_graph_engine()
    nodes = graph.get_all_nodes()
    if len(nodes) == 0:
        logger.info("Graph is empty on startup. Auto-ingesting synthetic sample dataset...")
        try:
            res = load_dataset_by_name(graph, "syndicate_network.json")
            logger.info(f"Auto-ingested sample data: {res['total_graph_nodes']} nodes, {res['total_graph_edges']} edges created.")
        except Exception as e:
            logger.error(f"Failed to auto-ingest sample dataset: {e}")
    else:
        logger.info(f"Graph loaded with {len(nodes)} pre-existing nodes.")

    # Initialize demo face storage
    try:
        f_storage = get_face_storage()
        if len(f_storage.list_identities()) == 0:
            seed_demo_face_database(f_storage)
            logger.info(f"Auto-seeded demo face database with {len(f_storage.list_identities())} identities.")
    except Exception as e:
        logger.warning(f"Could not auto-seed face database: {e}")

    # Initialize demo financial storage
    try:
        from app.core.financial_analytics import get_financial_storage, get_financial_service
        from app.core.financial_parser import FinancialParser
        f_store = get_financial_storage()
        if len(f_store.get_all_records()) == 0:
            demo_csv_path = os.path.join(settings.BASE_DIR, "sample_data", "demo_financial.csv")
            if os.path.exists(demo_csv_path):
                with open(demo_csv_path, "rb") as df:
                    records, _, _, _ = FinancialParser.parse_csv(df.read(), source_name="demo_financial.csv")
                    f_svc = get_financial_service()
                    f_svc.ingest_records_into_graph(records)
                    logger.info(f"Auto-ingested {len(records)} demo financial records.")
    except Exception as e:
        logger.warning(f"Could not auto-seed demo financial data: {e}")

    # Initialize keep-alive self-ping worker (prevents Render 15-minute idle spin-down)
    keep_alive_task = None
    try:
        from app.core.keep_alive import start_keep_alive_loop
        keep_alive_task = asyncio.create_task(start_keep_alive_loop())
    except Exception as e:
        logger.warning(f"Could not start keep-alive task: {e}")

    yield

    if keep_alive_task and not keep_alive_task.done():
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except (asyncio.CancelledError, Exception):
            pass

    audit_logger.log(
        action=AuditAction.SYSTEM_SHUTDOWN,
        status="SUCCESS",
        details={"project": settings.PROJECT_NAME},
    )


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-Assisted Criminal Network Analysis & Knowledge Graph System Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(users_router, prefix=settings.API_V1_STR)
app.include_router(audit_router, prefix=settings.API_V1_STR)
app.include_router(ingest_router, prefix=settings.API_V1_STR)
app.include_router(graph_router, prefix=settings.API_V1_STR)
app.include_router(analytics_router, prefix=settings.API_V1_STR)
app.include_router(alerts_router, prefix=settings.API_V1_STR)
app.include_router(search_router, prefix=settings.API_V1_STR)
app.include_router(search_router, prefix="")  # Support /search as well as /api/search
app.include_router(cdr_router, prefix=settings.API_V1_STR)
app.include_router(financial_router, prefix=settings.API_V1_STR)
app.include_router(timeline_router, prefix=settings.API_V1_STR)
app.include_router(risk_router, prefix=settings.API_V1_STR)
app.include_router(assistant_router, prefix=settings.API_V1_STR)
app.include_router(reports_router, prefix=settings.API_V1_STR)
app.include_router(cross_domain_router, prefix=settings.API_V1_STR)
app.include_router(evidence_router, prefix=settings.API_V1_STR)


# Static Files Setup
static_path = os.path.join(settings.BASE_DIR, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

sample_data_path = os.path.join(settings.BASE_DIR, "sample_data")
if os.path.exists(sample_data_path):
    app.mount("/sample_data", StaticFiles(directory=sample_data_path), name="sample_data")


@app.get("/health", tags=["system"])
@app.get("/api/health", tags=["system"])
def health_check():
    """System health check endpoint for monitoring probes."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "database": "active",
    }


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve single-page investigator dashboard frontend."""
    index_file = os.path.join(static_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Criminal Network Engine API running. Dashboard file not found."}

