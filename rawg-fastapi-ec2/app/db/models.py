from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, JSON, Float, Date
from app.db.session import Base

class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rawg_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    released: Mapped[Date | None]
    metacritic: Mapped[int | None]
    rating: Mapped[Float | None]
    ratings_count: Mapped[int | None]
    genres: Mapped[dict | None] = mapped_column(JSON)
    platforms: Mapped[dict | None] = mapped_column(JSON)
    developers: Mapped[dict | None] = mapped_column(JSON)
    publishers: Mapped[dict | None] = mapped_column(JSON)
    tags: Mapped[dict | None] = mapped_column(JSON)
