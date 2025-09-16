from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Cargar modelo y encoders
with open("xgb_model_prelaunch.pkl", "rb") as f:
    clf = pickle.load(f)
with open("mlb_gen_prelaunch.pkl", "rb") as f:
    mlb_gen = pickle.load(f)
with open("mlb_store_prelaunch.pkl", "rb") as f:
    mlb_store = pickle.load(f)
with open("mlb_plat_prelaunch.pkl", "rb") as f:
    mlb_plat = pickle.load(f)
with open("ohe_esrb_prelaunch.pkl", "rb") as f:
    ohe = pickle.load(f)

# Crear app
app = FastAPI(
    title="API Videojuegos",
    description="API para predecir si un videojuego será un éxito",
    version="1.0.0"
)

# Ruta principal
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de predicción de éxito de videojuegos"}

# Modelo de datos
class Game(BaseModel):
    rating: float = 0
    added: int = 0
    metacritic: float = 0
    genre_ids: list[int] = []
    store_ids: list[int] = []
    platform_ids: list[int] = []
    esrb_rating_id: int = -1

# Endpoint predict
@app.post("/predict")
def predict(game: Game):
    df = pd.DataFrame([game.dict()])
    df['_metacritic_norm'] = df['metacritic'] / 100
    df['_rating_norm'] = df['rating'] / 5
    gen_df = pd.DataFrame(mlb_gen.transform(df['genre_ids']), columns=[f"genre_{c}" for c in mlb_gen.classes_])
    store_df = pd.DataFrame(mlb_store.transform(df['store_ids']), columns=[f"store_{c}" for c in mlb_store.classes_])
    plat_df = pd.DataFrame(mlb_plat.transform(df['platform_ids']), columns=[f"plat_{c}" for c in mlb_plat.classes_])
    esrb_df = pd.DataFrame(ohe.transform(df[['esrb_rating_id']]), columns=[f"esrb_{c}" for c in ohe.categories_[0]])
    X = pd.concat([df[['rating','added','metacritic']], gen_df, store_df, plat_df, esrb_df], axis=1).fillna(0)
    proba = float(clf.predict_proba(X)[:,1][0])
    pred = int(proba >= 0.5)
    return {"predicted_success": pred, "success_probability": proba}
