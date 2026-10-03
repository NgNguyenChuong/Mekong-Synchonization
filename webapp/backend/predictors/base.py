"""
Interface chung cho moi predictor do man (mock hoac model ML that).

Muc dich: app.py chi goi qua interface nay, khong quan tam ben duoi la
cong thuc mock hay model that. Nho vay sau nay gan model that vao KHONG
can sua app.py/frontend, chi can doi PREDICTION_MODE va viet MLPredictor.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    salinity_ppt: float
    # Metadata di kem MOI prediction, de frontend/nguoi doc API khong bao
    # gio nham mock voi ket qua khoa hoc that.
    prediction_mode: str  # "mock" | "model"
    model_name: str | None = None
    model_version: str | None = None


class SalinityPredictor(ABC):
    """Interface cho 1 predictor do man."""

    @abstractmethod
    def predict(self, scope: str, h3_index: str, lat: float, lon: float, date: str) -> PredictionResult:
        """Du doan do man (ppt) cho 1 o H3 tai 1 ngay cu the."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Metadata mo ta predictor nay (mode, ten model, version...)."""
        raise NotImplementedError
