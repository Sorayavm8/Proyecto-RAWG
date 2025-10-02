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
from dotenv import load_dotenv
import os

# ----------------- INICIALIZAR APP -----------------
app = FastAPI(
    title="API Videojuegos",
    description="API para predicción, preguntas en texto y visualización",
    version="1.0.0"
)

# ----------------- CONEXIÓN A BASE DE DATOS -----------------
load_dotenv()  # Carga las variables del archivo .env

host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

if not all([host, user, password, database]):
    raise RuntimeError("Faltan variables de entorno de la base de datos en .env")


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
cols = ["name", "metacritic", "rating", "genre_ids"] # dejo columnas mas simples sin arrays y rinde peor
df = pd.read_sql_query(sql, engine)
df_copy = df[cols].head(200).reset_index(drop=True).copy() # limitar columnas al modelo a 100 porque si no da peor rendimiento o puede petar
table = df_copy.astype(str).fillna("")   # esto es lo que vamos a pasar al pipeline libre de nans y todo a string
assert table.shape[0] > 0 and table.shape[1] > 0, f"Tabla vacia: shape={table.shape}" # check de seguridad por si la tabla se queda vacia
#table = table.head(100).reset_index(drop=True)   # recorta mas si hace falta la tabla si da error
table.columns = [str(c) for c in table.columns] # parseo a strings extra por si nombres de columnas no lo son.

# TAPAS
model_name = "google/tapas-large-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

# --- preparar DF numérico paralelo ---
df_num = df_copy.copy()
df_num["metacritic"] = pd.to_numeric(df_num["metacritic"], errors="coerce")
df_num["rating"] = pd.to_numeric(df_num["rating"], errors="coerce")

def _choose_avg_column(query: str, coords: list[tuple[int,int]] | None) -> str:
    """
    Elige la columna a promediar:
    1) por palabra clave en la pregunta,
    2) si no, por las celdas seleccionadas por TAPAS,
    3) si no, default 'metacritic'.
    """
    ql = query.lower()
    if "metacritic" in ql:
        return "metacritic"
    if "rating" in ql:
        return "rating"

    if coords:
        cols_in_coords = [table.columns[c] for (_, c) in coords]
        # prioriza columnas numéricas si están en coords
        for cand in ("metacritic", "rating"):
            if cand in cols_in_coords:
                return cand

    return "metacritic"  # default sensato

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
    coords = pred_coords[0] or []

    if not coords and agg == "NONE":
        return {"answer": "No se ha encontrado respuesta", "aggregation": agg}

    if agg == "AVERAGE":
        target_col = _choose_avg_column(q.query, coords)

        # Si TAPAS marcó celdas de esa columna, promedia esas; si no, promedia toda la columna
        rows_for_target = [r for (r, c) in coords if table.columns[c] == target_col]
        if rows_for_target:
            values = df_num.loc[rows_for_target, target_col]
        else:
            values = df_num[target_col]

        value = round(values.mean(), 2)
        return {
            "query": q.query,
            "answer": value,
            "aggregation": agg,
            "column": target_col,
            "method": "tapas+heuristics"
        }

    elif agg == "COUNT":
        # cuenta filas ÚNICAS implicadas (evita sobreconteos)
        unique_rows = sorted({r for (r, c) in coords})
        value = len(unique_rows)
        return {
            "query": q.query,
            "answer": value,
            "aggregation": agg,
            "selected_rows_count": value,
            "method": "tapas (dedup rows)"
        }

    else:
        answers = [table.iat[r, c] for (r, c) in coords]
        return {"query": q.query, "answer": answers, "aggregation": agg, "method": "tapas"}
    
# ----------------- ENDPOINT /ask-visual -----------------
sql_visual = """SELECT g.*, COALESCE(ge.genre_ids, ARRAY[]::int[]) AS genre_ids
FROM public.games g
LEFT JOIN (
  SELECT gg.game_id, ARRAY_AGG(DISTINCT gg.genre_id ORDER BY gg.genre_id) AS genre_ids
  FROM public.game_genres gg GROUP BY gg.game_id
) ge ON ge.game_id = g.game_id
WHERE g.metacritic IS NOT NULL OR g.rating IS NOT NULL
ORDER BY g.game_id;"""

# Leer datos de la base
df_visual = pd.read_sql_query(sql_visual, engine)
df_genres = pd.read_sql("SELECT genre_id, name AS genre FROM public.genres", engine)

# Explode para genres y merge con nombres
df_exploded = df_visual.explode('genre_ids').merge(
    df_genres, left_on='genre_ids', right_on='genre_id', how='left'
)

# Copiar solo columnas relevantes
df_copy_visual = df_exploded[['name', 'rating', 'genre', "added"]].copy()

# Submuestrear filas para evitar problemas con modelos
df_copy_visual = df_copy_visual.head(100)

# Limpiar y convertir a strings
table_visual = df_copy_visual.astype(str).fillna("")
assert table_visual.shape[0] > 0 and table_visual.shape[1] > 0, f"Tabla vacia: shape={table_visual.shape}"
table_visual.columns = [str(c) for c in table_visual.columns]

# Cargar modelo TAPAS
model_name = "google/tapas-large-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

# Función para obtener filas y agregación
def get_rows_from_tapas(query):
    inputs = tokenizer(table=table_visual, queries=[query], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_coords, pred_aggs = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits, outputs.logits_aggregation
    )
    aggregation_map = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    agg = aggregation_map.get(pred_aggs[0], "NONE")
    rows = [r for (r, c) in pred_coords[0]]
    return rows, agg

# Función para generar gráfico
def plot_from_rows(query, rows, agg, target_col=None):
    plt.figure(figsize=(10,6))
    if not rows:
        plt.text(0.5, 0.5, "No se reconoció ninguna pregunta", ha="center", va="center")
        plt.axis("off")
    else:
        subset = df_copy_visual.iloc[rows]
        if agg == "AVERAGE":
            col = target_col or "rating"  # fallback si no se pasa
            data = subset.groupby("genre")[col].mean().sort_values(ascending=False)
            sns.barplot(x=data.values, y=data.index, palette="coolwarm")
            plt.xlabel(f"Average {col.capitalize()}"); plt.ylabel("Genre")
        elif agg == "COUNT":
            data = subset["genre"].value_counts()
            sns.barplot(x=data.values, y=data.index, palette="viridis")
            plt.xlabel("Number of Games"); plt.ylabel("Genre")
        else:
            col = target_col or "rating"
            sns.barplot(x=subset[col], y=subset["genre"], palette="magma")
            plt.xlabel(col.capitalize()); plt.ylabel("Genre")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")


# Modelo Pydantic para request
class QuestionVisual(BaseModel):
    query: str | list[str]

# Endpoint FastAPI
@app.post("/ask-visual")
def ask_visual(q: QuestionVisual):
    queries = q.query if isinstance(q.query, list) else [q.query]
    rows, agg = get_rows_from_tapas(queries[0])
    return plot_from_rows(queries[0], rows, agg)

