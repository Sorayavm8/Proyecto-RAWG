from fastapi import APIRouter
from .routes import predict, qa

router = APIRouter()
router.include_router(predict.router, prefix="/ml", tags=["ml"])
router.include_router(qa.router, prefix="/qa", tags=["qa"])
