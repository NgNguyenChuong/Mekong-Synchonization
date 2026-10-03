"""
MLPredictor - predictor dung model ML that (CHUA CO MODEL).

Day la SKELETON, khong fabricate ket qua. Khi chua co model da train that
(xem src/training/train.py va docs/model_training.md), predictor nay se
raise loi ro rang thay vi tra ve so lieu gia.

Khi da co model that, can implement:
1. Load model + preprocessing pipeline (vd: sklearn Pipeline da fit,
   luu cung 1 file .joblib de dam bao transform giong het luc train).
2. Load feature dataset (tu src/ pipeline: DYNAMIC_MERGE.csv,
   STATIC_MERGED.csv, PERIODIC_MERGED.csv) de tra cuu feature theo
   (h3_index, date).
3. predict(): build feature vector dung thu tu/ten cot nhu luc train,
   goi model.predict(), tra ve PredictionResult voi prediction_mode="model".

KHONG duoc preprocess khac giua training va inference - nen serialize ca
preprocessing pipeline (vd sklearn ColumnTransformer) cung model.
"""
import os

from predictors.base import PredictionResult, SalinityPredictor


class ModelNotAvailableError(RuntimeError):
    pass


class MLPredictor(SalinityPredictor):
    def __init__(self, model_dir: str, model_version: str = "unknown"):
        self.model_dir = model_dir
        self.model_version = model_version
        self._model = None  # se duoc load trong _load() khi co model that

        if not os.path.isdir(model_dir):
            raise ModelNotAvailableError(
                f"BLOCKED BY REAL DATA: khong tim thay thu muc model '{model_dir}'. "
                "Can chay src/training/train.py de tao model truoc, hoac tro "
                "PREDICTION_MODEL_DIR toi thu muc chua model da train."
            )

        # TODO: khi co model that, load model + preprocessing pipeline o day.
        # self._model = joblib.load(os.path.join(model_dir, "model.joblib"))
        raise ModelNotAvailableError(
            "BLOCKED BY REAL DATA: MLPredictor chua duoc implement day du vi "
            "chua co model ML da train. Xem docstring file nay va "
            "docs/model_training.md de biet cac buoc con thieu."
        )

    def predict(self, scope: str, h3_index: str, lat: float, lon: float, date: str) -> PredictionResult:
        raise ModelNotAvailableError("MLPredictor chua co model that de inference.")

    @property
    def metadata(self) -> dict:
        return {
            "prediction_mode": "model",
            "model_name": "unset",
            "model_version": self.model_version,
        }
