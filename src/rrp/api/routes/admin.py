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
    new_mape_raw = result.get("new_mape")
    new_mape: float | None
    if new_mape_raw is None:
        new_mape = None
    else:
        new_mape = float(new_mape_raw)  # type: ignore[arg-type]
    return RetrainResponse(
        promoted=bool(result.get("promoted", False)),
        version=str(result.get("version")) if result.get("version") else None,
        new_mape=new_mape,
        message=(
            f"New model promoted (version {result.get('version')})."
            if result.get("promoted")
            else "Retrain completed — current champion retained."
        ),
    )
