from fastapi import APIRouter
from .data_controller import router as data_router
from .vector_controller import router as vector_router

router = APIRouter(prefix="/v1")

router.include_router(data_router)
router.include_router(vector_router)