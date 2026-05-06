from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from rrp.api.schemas import RetrainResponse
from rrp.core.db import get_db
from rrp.forecasting.trainer import run_nightly_retrain

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/retrain", response_model=RetrainResponse, summary="Manually trigger nightly retrain")
def trigger_retrain(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RetrainResponse:
    result = run_nightly_retrain(db)
    return RetrainResponse(
        promoted=bool(result.get("promoted", False)),
        version=str(result.get("version")) if result.get("version") else None,
        new_mape=float(result["new_mape"]) if result.get("new_mape") is not None else None,
        message=(
            f"New model promoted (version {result.get('version')})."
            if result.get("promoted")
            else "Retrain completed — current champion retained."
        ),
    )
