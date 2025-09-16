from pydantic import BaseModel, Field
from typing import Optional

class GameFeatures(BaseModel):
    name: str
    metacritic: Optional[int] = None
    rating: Optional[float] = None
    ratings_count: Optional[int] = None
    genres: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    developers: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

class PredictOut(BaseModel):
    success: int
    probability: float
