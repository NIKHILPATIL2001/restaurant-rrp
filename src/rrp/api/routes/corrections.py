from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from rrp.api.schemas import CorrectionRequest, CorrectionResponse
from rrp.core.db import get_db
from rrp.domain.models import CorrectionScope, ReasonCode
from rrp.feedback.service import apply_correction

router = APIRouter(prefix="/corrections", tags=["corrections"])


@router.post("", response_model=CorrectionResponse, summary="Submit a manager correction")
def submit_correction(
    req: CorrectionRequest,
    db: Session = Depends(get_db),
) -> CorrectionResponse:
    try:
        scope = CorrectionScope(req.scope)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid scope: {req.scope}") from exc

    try:
        reason = ReasonCode(req.reason_code)
    except ValueError:
        reason = ReasonCode.unknown

    result = apply_correction(
        scope=scope,
        scope_id=req.scope_id,
        predicted=req.predicted,
        actual=req.actual,
        reason_code=reason,
        note=req.note,
        correction_date=req.date,
        hour=req.hour,
        db=db,
    )

    return CorrectionResponse(
        correction_id=int(result.get("correction_id") or 0),
        quarantined=result.get("quarantined", False),
        online_updated=result.get("online_updated", False),
        message=(
            "Correction quarantined for review — delta exceeds plausibility threshold."
            if result.get("quarantined")
            else "Correction applied. Online learner updated."
        ),
    )
