from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

CANNED = {
    "desarrollador con la mejor puntuación": (
        "SELECT jsonb_array_elements_text(developers) AS developer, "
        "AVG(metacritic) AS avg_meta "
        "FROM games GROUP BY developer ORDER BY avg_meta DESC LIMIT 1;"
    ),
    "top 10 géneros por número de juegos": (
        "SELECT jsonb_array_elements_text(genres) AS genre, "
        "COUNT(*) AS n "
        "FROM games GROUP BY genre ORDER BY n DESC LIMIT 10;"
    ),
}

async def question_to_sql(question: str) -> str:
    q = question.lower().strip()
    for k, v in CANNED.items():
        if k in q:
            return v
    return "SELECT COUNT(*) AS total_games FROM games;"

async def run_sql(sql: str, db: AsyncSession):
    res = await db.execute(text(sql))
    rows = res.fetchall()
    cols = res.keys()
    return [dict(zip(cols, r)) for r in rows]
