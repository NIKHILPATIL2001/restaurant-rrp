from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RRP_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rrp:rrp@localhost:5432/rrp"
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    env: str = "development"

    # Retrain schedule (cron expression, default: 2 AM daily)
    retrain_cron_hour: int = 2
    retrain_cron_minute: int = 0

    # Simulation
    warmup_days: int = 30

    # Convergence thresholds
    drift_mape_multiplier: float = 1.5
    champion_eval_days: int = 7

    # Correction plausibility
    quarantine_delta_ratio: float = 5.0

    # Clipping bounds
    clip_lower_factor: float = 0.5
    clip_upper_factor: float = 2.0

    # Cold-start regimes
    sparse_threshold_days: int = 28

    # Online residual learner — see src/rrp/forecasting/online.py for the
    # design rationale behind each of these defaults.
    #
    # The online layer sits on top of a LightGBM baseline that already lands
    # at ~5-7% MAPE on the synthetic dataset. With manager-noise of ±5% on
    # each correction and only a handful of corrections per (weekday, hour)
    # cell over a 60-day window, the online layer's signal-to-noise is low.
    # Defaults are tuned to "do nothing unless we are sure" so the layer
    # cannot regress the baseline on noisy days.
    online_alpha: float = 0.06          # half-life ≈ 11.2 corrections
    online_warmup_n: int = 12           # ramp to full strength only after evidence
    online_significance_k: float = 1.0  # |EWMA| must exceed 1×MAD before emitting
    online_ridge_lam: float = 50.0      # heavy regularisation; ridge rarely moves
    online_ridge_warmup_n: int = 25
    online_residual_clamp: float = 0.20  # max |residual| as fraction of |predicted|
    # Hard cap on the *applied* correction at predict time, expressed as a
    # fraction of the baseline prediction at that hour. Final safety net:
    # even if EWMA + ridge sum to a large number, the correction visible
    # downstream is bounded. Also bounds the worst-case MAPE regression the
    # online layer can introduce.
    online_predict_cap_factor: float = 0.08

    # Convergence test — corrected MAPE on the late window must improve over
    # the late-window baseline by at least this much, AND the worst-day
    # corrected MAPE must not exceed the late-window baseline mean by more
    # than `convergence_no_regression_factor`.
    convergence_window_days: int = 7
    convergence_no_regression_factor: float = 1.10

    @property
    def models_path(self) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir

    @property
    def data_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


settings = Settings()
