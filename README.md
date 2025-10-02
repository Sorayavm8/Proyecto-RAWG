# API Videojuegos - Proyecto RAWG

Esta API es un ejemplo de proyecto que combina **Machine Learning**, **IA para preguntas en tablas** y **visualización de datos** sobre videojuegos. Está diseñada como portfolio para mostrar habilidades en Python, FastAPI, análisis de datos y modelos NLP.

---

## Descripción del proyecto

La API permite:

1. **Predicción de éxito de un juego** usando un modelo de ML (`/predict`).  
2. **Responder preguntas sobre datos de juegos** en formato de tabla (`/ask-text`).  
3. **Generar visualizaciones automáticas** a partir de preguntas sobre los datos (`/ask-visual`).

Se ha diseñado para que funcione con datos de ejemplo, sin necesidad de exponer credenciales de base de datos reales.


## Endpoints

| Endpoint       | Método | Descripción |
|----------------|--------|-------------|
| `/`            | GET    | Mensaje de bienvenida |
| `/predict`     | POST   | Predice la probabilidad de éxito de un juego |
| `/ask-text`    | POST   | Responde preguntas sobre los datos en formato tabla |
| `/ask-visual`  | POST   | Genera gráficos basados en consultas sobre los datos |

## Flujo de trabajo y preparación de datos

1. **Cargar modelos y encoders**:
   - `xgb_model_prelaunch.pkl` → modelo de XGBoost
   - `mlb_gen_prelaunch.pkl`, `mlb_store_prelaunch.pkl`, `mlb_plat_prelaunch.pkl` → codificadores multi-label
   - `ohe_esrb_prelaunch.pkl` → codificador one-hot para ESRB

2. **Preparar tablas** para TAPAS:
   - Extraer columnas clave: `name`, `rating`, `metacritic`, `genre_ids`.
   - Convertir todo a string y rellenar `NaN` para que TAPAS funcione correctamente.

3. **Preprocesar datos numéricos**:
   - Convertir columnas `rating` y `metacritic` a valores numéricos para operaciones de agregación (`AVERAGE`, `SUM`).

4. **Generación de visualizaciones**:
   - Se submuestrean filas para evitar problemas de rendimiento.
   - Se utilizan `seaborn` y `matplotlib` para generar gráficos en función de la pregunta.
   - Las respuestas se devuelven como imágenes PNG (`StreamingResponse`).
   
## Cómo correr
Instalar dependencias: pip install -r requirements.txt

Ejecutar la API: uvicorn app:app --reload


## Requisitos
Python 3.11+
fastapi
uvicorn
pandas
matplotlib
seaborn
torch
transformers
sqlalchemy
psycopg2-binary
python-dotenv

## Ejemplos de uso

### `/predict`
Request:
```json
{
  "rating": 4.5,
  "added": 1200,
  "metacritic": 85,
  "genre_ids": [1, 5],
  "store_ids": [2, 3],
  "platform_ids": [4],
  "esrb_rating_id": 2
}
{
  "predicted_success": 1,
  "success_probability": 0.78
}
{
  "query": "Average metacritic of games"
}
{
  "query": "Average metacritic of games",
  "answer": 72.5,
  "aggregation": "AVERAGE",
  "column": "metacritic",
  "method": "tapas+heuristics"
}
{
  "query": "Count of games per genre"
}
Response: Devuelve un gráfico PNG con la distribución de juegos por género.