import os, joblib
from typing import Any, Dict

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
PREPROC_PATH = os.getenv("PREPROC_PATH", "models/preprocess.joblib")

class Predictor:
    def __init__(self):
        self.model = None
        self.pre = None

    def load(self):
        if self.model is None and os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        if self.pre is None and os.path.exists(PREPROC_PATH):
            self.pre = joblib.load(PREPROC_PATH)

    def predict(self, payload: Dict[str, Any]):
        self.load()
        if self.model is None or self.pre is None:
            prob = float(min(0.99, max(0.01, (payload.get("rating") or 0) / 5)))
            return (1 if prob > 0.6 else 0), prob
        import pandas as pd
        X = pd.DataFrame([payload])
        Xp = self.pre.transform(X)
        proba = self.model.predict_proba(Xp)[0, 1]
        label = int(proba >= 0.5)
        return label, float(proba)

predictor = Predictor()
