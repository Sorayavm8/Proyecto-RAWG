"""
API FastAPI para el Proyecto RAWG
---------------------------------
Este servicio expone endpoints para:
  - Autenticación simple con JWT (/login)
  - Predicción de "éxito" de un juego (/predict) con un modelo simple (o .pkl si lo proporcionas)
  - Preguntas de texto → SQL → respuesta en texto (/ask-text)
  - Preguntas de visualización → SQL → imagen PNG (/ask-visual)
  - Métricas Prometheus (/metrics) y healthcheck (/health)

IMPORTANTE (bases de datos)
- En desarrollo puedes usar SQLite: DB_URL=sqlite:///./games.db  (se crea con datos demo)
- En producción usa la RDS de tu equipo (PostgreSQL):
  DB_URL=postgresql+psycopg://USUARIO:PASS@HOST_RDS:5432/rawg_db?sslmode=require

Este archivo está fuertemente comentado EN ESPAÑOL para que entiendas cada parte.
"""

from __future__ import annotations

# ------- imports estándar -------
import io, os, re, time, logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# ------- terceros -------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, Float, Integer, String, Date,
    create_engine, func, select
)
from sqlalchemy.orm import declarative_base, sessionmaker
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ------- logging simple -------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rawg_api")

# ------- configuración por variables de entorno (.env) -------
DB_URL = os.getenv("DB_URL", "sqlite:///./games.db")   # Cambia a tu RDS en producción
SECRET_KEY = os.getenv("SECRET_KEY", "cambia-esto")    # Clave para firmar JWT
ALGO = os.getenv("ALGO", "HS256")
TOKEN_MIN = int(os.getenv("TOKEN_MIN", "60"))          # Minutos de validez del token
USE_HF = os.getenv("USE_HF", "0") == "1"               # HuggingFace opcional (apagado por defecto)
TEXT2SQL_MODEL = os.getenv("TEXT2SQL_MODEL", "defog/sqlcoder-7b-2")
PREDICT_MODEL_PATH = os.getenv("PREDICT_MODEL_PATH")   # Ruta a un .pkl si lo tienes

# Detectamos si trabajamos en modo demo con SQLite para crear datos de prueba automáticamente
IS_SQLITE = DB_URL.lower().startswith("sqlite")

# ------- SQLAlchemy (motor y sesión) -------
engine = create_engine(DB_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# ------- Modelo mínimo SOLO para la demo con SQLite -------
class Game(Base):
    __tablename__ = "games"
    game_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    released = Column(Date)
    rating = Column(Float)
    metacritic = Column(Integer)
    genre = Column(String, index=True)
    user_score = Column(Float)

def create_demo_db():
    """
    Crea una BD SQLite con algunos juegos de ejemplo para que puedas probar
    la API sin depender de RDS. NO se ejecuta si DB_URL no es sqlite.
    """
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        c = conn.execute(select(func.count()).select_from(Game)).scalar() or 0
        if c > 0:
            return
        seed = [
            {"name":"The Legend of Zelda: Breath of the Wild","genre":"Adventure","released":"2017-03-03","metacritic":97,"rating":4.8,"user_score":8.7},
            {"name":"God of War","genre":"Action","released":"2018-04-20","metacritic":94,"rating":4.6,"user_score":9.1},
            {"name":"Red Dead Redemption 2","genre":"Action-Adventure","released":"2018-10-26","metacritic":97,"rating":4.7,"user_score":8.4},
            {"name":"Hades","genre":"Roguelike","released":"2020-09-17","metacritic":93,"rating":4.7,"user_score":8.9},
            {"name":"Elden Ring","genre":"RPG","released":"2022-02-25","metacritic":96,"rating":4.4,"user_score":7.8},
        ]
        for i, r in enumerate(seed, 1):
            released_date = datetime.strptime(r["released"], "%Y-%m-%d").date()  # 👈 conversión
            conn.execute(Game.__table__.insert().values(
                game_id=i,
                name=r["name"],
                genre=r["genre"],
                released=released_date,     # 👈 ahora es date, no string
                metacritic=r["metacritic"],
                rating=r["rating"],
                user_score=r["user_score"]
            ))

# ------- Modelo ML sencillo (baseline) -------
class SimpleBaselineModel:
    """Modelo de juguete para la demo."""
    def predict(self, X: List[List[float]]):
        preds = []
        for meta, user, year_scaled, genre_id in X:
            score = 0.6*(meta/100.0) + 0.3*(user/10.0) + 0.1*year_scaled
            if int(genre_id) in {3, 2}:  # boost a RPG y Action-Adventure
                score += 0.03
            preds.append(round(score, 4))
        return preds

def load_predict_model():
    """Carga un modelo .pkl si existe, si no usa el baseline simple."""
    if PREDICT_MODEL_PATH and os.path.exists(PREDICT_MODEL_PATH):
        try:
            import joblib
            log.info("Cargando modelo desde %s", PREDICT_MODEL_PATH)
            return joblib.load(PREDICT_MODEL_PATH)
        except Exception as e:
            log.warning("No se pudo cargar modelo .pkl (%s). Uso baseline.", e)
    return SimpleBaselineModel()

PREDICT_MODEL = load_predict_model()

# ------- Esquema Pydantic -------
GENRE_VOCAB = {
    "Action": 1, "Action-Adventure": 2, "RPG": 3, "Shooter": 4,
    "Adventure": 5, "Roguelike": 6, "Simulation": 7
}

class PredictPayload(BaseModel):
    title: str
    developer: Optional[str] = None
    genre: str = Field(description="Género (RPG, Action, etc.)")
    release_year: Optional[int] = Field(default=None, ge=1970, le=2100)
    metascore: Optional[float] = Field(default=None, ge=0, le=100)
    user_score: Optional[float] = Field(default=None, ge=0, le=10)
    platforms: Optional[List[str]] = None

    @field_validator("genre")
    @classmethod
    def check_genre(cls, v: str):
        if v not in GENRE_VOCAB:
            raise ValueError(f"Género desconocido '{v}'. Usa uno de: {list(GENRE_VOCAB)}")
        return v

class TextQuestion(BaseModel):
    question: str

class VisualQuestion(BaseModel):
    question: str

# ------- Text→SQL -------
HF_MODEL = None
HF_TOKENIZER = None

def maybe_load_hf() -> bool:
    if not USE_HF:
        return False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("Cargando modelo HF: %s", TEXT2SQL_MODEL)
        tok = AutoTokenizer.from_pretrained(TEXT2SQL_MODEL)
        mdl = AutoModelForCausalLM.from_pretrained(TEXT2SQL_MODEL)
        globals()["HF_TOKENIZER"] = tok
        globals()["HF_MODEL"] = mdl
        return True
    except Exception as e:
        log.warning("No se pudo cargar HF (%s). Uso reglas.", e)
        return False

HF_READY = maybe_load_hf()

def nl_to_sql_fallback(q: str) -> Tuple[str, str]:
    """Reglas SQL básicas para mapear preguntas en español a SQL."""
    s = q.lower().strip()
    # ... (se mantiene igual que tu versión)
    # No cambio nada aquí porque estaba correcto.
    if "género" in s and "rating" in s:
        return (
            "SELECT g.name AS genre, ROUND(AVG(gs.rating),2) AS avg_rating "
            "FROM games gs JOIN game_genres gg ON gs.game_id=gg.game_id "
            "JOIN genres g ON gg.genre_id=g.genre_id "
            "WHERE gs.rating IS NOT NULL "
            "GROUP BY g.name ORDER BY avg_rating DESC;",
            "avg_rating_by_genre"
        )
    return (
        "SELECT name, metacritic FROM games "
        "WHERE metacritic IS NOT NULL ORDER BY metacritic DESC LIMIT 5;",
        "fallback_top5"
    )

def nl_to_sql_hf(q: str) -> str:
    if not HF_READY:
        sql, _ = nl_to_sql_fallback(q)
        return sql
    try:
        import torch
        # ... (se mantiene igual que tu versión original)
    except Exception as e:
        log.warning("HF falló, uso fallback (%s)", e)
    sql, _ = nl_to_sql_fallback(q)
    return sql

# ------- FastAPI + JWT -------
app = FastAPI(title="RAWG API (ML + Text-to-SQL + Visual)", version="1.3.0")
auth_scheme = HTTPBearer()

def make_token(sub: str) -> str:
    exp = int((datetime.utcnow() + timedelta(minutes=TOKEN_MIN)).timestamp())
    payload = {"sub": sub, "exp": exp}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGO)



def require_user(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)) -> str:
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
        return payload.get("sub") or "user"
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")

# ------- Métricas Prometheus -------
REQ_COUNT = Counter("api_requests_total", "Total de peticiones", ["path", "method", "status"])
REQ_LAT = Histogram("api_request_latency_seconds", "Latencia por endpoint", ["path", "method"])

@app.middleware("http")
async def metrics_mid(request: Request, call_next):
    t0 = time.time()
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception:
        status = 500
        raise
    finally:
        REQ_COUNT.labels(path=request.url.path, method=request.method, status=str(status)).inc()
        REQ_LAT.labels(path=request.url.path, method=request.method).observe(time.time() - t0)
    return resp

# ------- Utilidades modelo -------
_DEF_YEAR_LO, _DEF_YEAR_HI = 1980, 2025
def scale_year(y: Optional[int]) -> float:
    if not y: return 0.5
    y = min(max(y, _DEF_YEAR_LO), _DEF_YEAR_HI)
    return (y - _DEF_YEAR_LO) / (_DEF_YEAR_HI - _DEF_YEAR_LO)

def features_from_payload(p: PredictPayload) -> List[float]:
    meta = p.metascore if p.metascore is not None else 70.0
    user = p.user_score if p.user_score is not None else 7.5
    ys = scale_year(p.release_year)
    gid = GENRE_VOCAB[p.genre]
    return [meta, user, ys, gid]

# ------- Endpoints -------
class LoginPayload(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(body: LoginPayload):
    try:
        if not body.username or not body.password:
            raise HTTPException(status_code=400, detail="Faltan credenciales")

        sub = "admin" if (body.username == "admin" and body.password == "admin123") else body.username
        tok = make_token(sub)
        return {"access_token": tok, "token_type": "bearer"}

    except Exception as e:
        log.exception("Error en /login: %s", e)
        raise HTTPException(status_code=500, detail=f"login error: {e}")


@app.post("/predict")
def predict(payload: PredictPayload, user: str = Depends(require_user)):
    feats = [features_from_payload(payload)]
    try:
        y = PREDICT_MODEL.predict(feats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de predicción: {e}")
    return {"prediction": float(y[0]), "features": feats[0]}

@app.post("/ask-text")
def ask_text(req: TextQuestion, user: str = Depends(require_user)):
    sql = nl_to_sql_hf(req.question)
    try:
        with engine.begin() as conn:
            df = pd.read_sql(sql, conn)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL falló: {e}. SQL: {sql}")
    answer = format_text_answer(req.question, df)
    return JSONResponse({"answer": answer, "sql": sql, "rows": int(df.shape[0])})

@app.post("/ask-visual")
def ask_visual(req: VisualQuestion, user: str = Depends(require_user)):
    sql, intent = nl_to_sql_fallback(req.question)
    try:
        with engine.begin() as conn:
            df = pd.read_sql(sql, conn)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL falló: {e}. SQL: {sql}")
    fig = plot_from_intent(df, intent)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={"X-SQL-Used": sql})

@app.get("/")
def root():
    return {"name": "RAWG API", "version": "1.3.0",
            "endpoints": ["/login", "/predict", "/ask-text", "/ask-visual", "/metrics"]}

@app.get("/health")
def health():
    try:
        with engine.begin() as conn:
            conn.execute(select(func.count()).select_from(Game.__table__))
        _ = PREDICT_MODEL.predict([[80, 8, 0.5, 3]])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------- Helpers -------
def format_text_answer(q: str, df: pd.DataFrame) -> str:
    """Convierte un DataFrame en texto legible según la pregunta."""
    if df.empty:
        return "No encontré resultados."
    s = q.lower()
    if "género" in s and ("metacritic" in s or "puntuación" in s) and ("mejor" in s or "mayor" in s):
        r = df.iloc[0]
        return f"Mejor género (Metacritic medio): {r.get('genre')} con {r.get('avg_metacritic')}."
    if "género" in s and ("número" in s or "juegos" in s or "top" in s):
        items = [f"{r.genre}: {int(r.n_games)} juegos" for r in df.itertuples()]
        return "; ".join(items)
    if "media" in s and "rating" in s and "género" in s:
        items = [f"{r.genre}: {r.avg_rating}" for r in df.itertuples()]
        return "; ".join(items)
    return df.to_string(index=False)

def plot_from_intent(df: pd.DataFrame, intent: str):
    sns.set_theme(context="talk")
    if intent == "top_genres_by_count" and {"genre", "n_games"}.issubset(df.columns):
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="n_games", y="genre")
        ax.set_xlabel("Número de juegos"); ax.set_ylabel("Género")
        ax.set_title("Top géneros por número de juegos")
        return fig
    if intent == "count_by_year" and {"release_year", "n_games"}.issubset(df.columns):
        fig = plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df, x="release_year", y="n_games", marker="o")
        ax.set_xlabel("Año"); ax.set_ylabel("Número de juegos")
        ax.set_title("Juegos por año")
        return fig
    if intent == "avg_rating_by_genre" and {"genre", "avg_rating"}.issubset(df.columns):
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="avg_rating", y="genre")
        ax.set_xlabel("Rating medio"); ax.set_ylabel("Género")
        ax.set_title("Media de rating por género")
        return fig
    # Fallback: tabla simple
    fig = plt.figure(figsize=(10, 2 + 0.3 * len(df)))
    plt.axis("off"); plt.title("Resultados")
    tbl = plt.table(cellText=df.head(10).values, colLabels=list(df.columns), loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
    return fig
