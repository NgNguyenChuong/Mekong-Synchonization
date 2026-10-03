"""
Config trung tam cho webapp/backend.

Web chi dung du lieu that: predictor duy nhat la MLPredictor. Khi chua co model
da train (src/training/train.py), backend dung ngay luc khoi dong voi loi
ModelNotAvailableError ro rang thay vi tra so lieu gia (che do mock da bo).

PREDICTION_MODEL_DIR: thu muc chua model da train.
"""
import os

PREDICTION_MODEL_DIR = os.getenv("PREDICTION_MODEL_DIR", os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts", "experiments", "current"
))


def get_predictor():
    """Factory tra ve predictor dung model that."""
    from predictors.ml_predictor import MLPredictor
    return MLPredictor(model_dir=PREDICTION_MODEL_DIR)
