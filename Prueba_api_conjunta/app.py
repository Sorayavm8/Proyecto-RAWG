from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import torch
from sqlalchemy import create_engine
from transformers import TapasTokenizer, TapasForQuestionAnswering

# ----------------- INICIALIZAR APP -----------------
app = FastAPI(
    title="API Videojuegos",
    description="API para predicción, preguntas en texto y visualización",
    version="1.0.0"
)

# ----------------- CONEXIÓN A BASE DE DATOS -----------------
host = "rawg-db.c5gics8qchki.eu-north-1.rds.amazonaws.com"
user = "rawg_admin"
password = "Patata123!"
database = "rawg_db"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}/{database}")

# ----------------- ENDPOINT ROOT -----------------
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Videojuegos 🚀"}

# ----------------- ENDPOINT /predict -----------------
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

class Game(BaseModel):
    rating: float = 0
    added: int = 0
    metacritic: float = 0
    genre_ids: list[int] = []
    store_ids: list[int] = []
    platform_ids: list[int] = []
    esrb_rating_id: int = -1

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

# ----------------- ENDPOINT /ask-text -----------------
sql = """SELECT
  g.*,
  COALESCE(s.store_ids, ARRAY[]::int[]) AS store_ids,
  COALESCE(ge.genre_ids, ARRAY[]::int[]) AS genre_ids,
  COALESCE(p.platform_ids, ARRAY[]::int[]) AS platform_ids
FROM public.games g
LEFT JOIN (
  SELECT gs.game_id, ARRAY_AGG(DISTINCT gs.store_id ORDER BY gs.store_id) AS store_ids
  FROM public.game_stores gs GROUP BY gs.game_id
) s ON s.game_id = g.game_id
LEFT JOIN (
  SELECT gg.game_id, ARRAY_AGG(DISTINCT gg.genre_id ORDER BY gg.genre_id) AS genre_ids
  FROM public.game_genres gg GROUP BY gg.game_id
) ge ON ge.game_id = g.game_id
LEFT JOIN (
  SELECT gp.game_id, ARRAY_AGG(DISTINCT gp.platform_id ORDER BY gp.platform_id) AS platform_ids
  FROM public.game_platforms gp GROUP BY gp.game_id
) p ON p.game_id = g.game_id
ORDER BY g.game_id;
"""

df = pd.read_sql_query(sql, engine)
df_copy = df[["name", "metacritic", "rating"]].head(200).reset_index(drop=True)
table = df_copy.astype(str).fillna("")
table.columns = [str(c) for c in table.columns]

# TAPAS
model_name = "google/tapas-large-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

class QuestionText(BaseModel):
    query: str

@app.post("/ask-text")
def ask_text(q: QuestionText):
    inputs = tokenizer(table=table, queries=[q.query], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_coords, pred_aggs = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits, outputs.logits_aggregation
    )
    aggregation_map = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    agg = aggregation_map.get(pred_aggs[0], "UNKNOWN")
    rows = [r for (r, c) in pred_coords[0]]

    if not rows:
        return {"answer": "No se ha encontrado respuesta", "aggregation": agg}

    if agg == "AVERAGE":
        value = round(pd.to_numeric(df_copy.loc[rows, "rating"], errors="coerce").mean(), 2)
        return {"answer": value, "aggregation": agg}
    elif agg == "COUNT":
        value = len(rows)
        return {"answer": value, "aggregation": agg}
    else:
        answers = [table.iat[r, c] for (r, c) in pred_coords[0]]
        return {"answer": answers, "aggregation": agg}

# ----------------- ENDPOINT /ask-visual -----------------
sql_visual = """SELECT g.*, COALESCE(ge.genre_ids, ARRAY[]::int[]) AS genre_ids
FROM public.games g
LEFT JOIN (
  SELECT gg.game_id, ARRAY_AGG(DISTINCT gg.genre_id ORDER BY gg.genre_id) AS genre_ids
  FROM public.game_genres gg GROUP BY gg.game_id
) ge ON ge.game_id = g.game_id
WHERE g.metacritic IS NOT NULL OR g.rating IS NOT NULL
ORDER BY g.game_id;"""

df_visual = pd.read_sql_query(sql_visual, engine)
df_genres = pd.read_sql("SELECT genre_id, name AS genre FROM public.genres", engine)
df_exploded = df_visual.explode('genre_ids').merge(df_genres, left_on='genre_ids', right_on='genre_id', how='left')
df_copy_visual = df_exploded[['name', 'rating', 'genre']].copy()

class QuestionVisual(BaseModel):
    query: str | list[str]

@app.post("/ask-visual")
def ask_visual(q: QuestionVisual):
    queries = q.query if isinstance(q.query, list) else [q.query]
    plt.figure(figsize=(10,6))
    handled = False

    for query in queries:
        query_lower = query.lower()
        if "top 10" in query_lower and "género" in query_lower:
            data = df_copy_visual["genre"].value_counts().head(10)
            sns.barplot(x=data.values, y=data.index, palette="viridis")
            plt.xlabel("Número de juegos"); plt.ylabel("Género"); plt.title("Top 10 géneros por número de juegos")
            handled = True; break
        elif "rating promedio por género" in query_lower:
            data = df_copy_visual.groupby("genre")["rating"].mean().sort_values(ascending=False)
            sns.barplot(x=data.values, y=data.index, palette="coolwarm")
            plt.xlabel("Rating promedio"); plt.ylabel("Género"); plt.title("Rating promedio por género")
            handled = True; break

    if not handled:
        plt.text(0.5, 0.5, "No se reconoció ninguna pregunta", ha="center", va="center")
        plt.axis("off")

    buf = io.BytesIO()
    plt.tight_layout(); plt.savefig(buf, format="png")
    buf.seek(0); plt.close()
    return StreamingResponse(buf, media_type="image/png")
