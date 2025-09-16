from fastapi import APIRouter
from app.schemas.predict import GameFeatures, PredictOut
from app.services.predictor import predictor

router = APIRouter()

@router.post("/predict", response_model=PredictOut)
async def predict_game(data: GameFeatures):
    label, proba = predictor.predict(data.model_dump())
    return PredictOut(success=label, probability=proba)
