from fastapi import APIRouter
from .data_controller import router as data_router


router = APIRouter(prefix="/v1")

router.include_router(data_router)
