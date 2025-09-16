from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.services.text2sql import question_to_sql, run_sql
from app.services.visuals import plot_bar, to_b64

router = APIRouter()

@router.get("/ask-text")
async def ask_text(q: str, db: AsyncSession = Depends(get_db)):
    sql = await question_to_sql(q)
    rows = await run_sql(sql, db)
    return {"question": q, "sql": sql, "result": rows}

@router.get("/ask-visual")
async def ask_visual(q: str, db: AsyncSession = Depends(get_db)):
    sql = await question_to_sql(q)
    rows = await run_sql(sql, db)
    if not rows:
        raise HTTPException(404, "Sin resultados")
    first = rows[0]
    if len(first) != 2:
        raise HTTPException(400, "La consulta no es visualizable automáticamente")
    labels = [str(r[list(r.keys())[0]]) for r in rows]
    values = [float(r[list(r.keys())[1]]) for r in rows]
    img = plot_bar(labels, values)
    return {"question": q, "sql": sql, "image_base64_png": to_b64(img)}
