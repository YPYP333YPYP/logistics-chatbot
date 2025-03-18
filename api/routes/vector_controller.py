from fastapi import APIRouter

router = APIRouter(
    prefix="/api/vector",
    tags=["vector"],
    responses={404: {"description": "Not found"}}
)