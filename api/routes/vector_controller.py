from fastapi import APIRouter, Depends, Body, HTTPException

from schema.vector_schema import RebuildModel
from service.vector_service import VectorService

router = APIRouter(
    prefix="/api/vector",
    tags=["vector"],
    responses={404: {"description": "Not found"}}
)


@router.post("/build")
async def build_vector_store(
        rebuild_model: RebuildModel = Body(...),
        vector_service: VectorService = Depends(VectorService)
):
    """
    벡터 스토어 구축/재구축 API

    Args:
        rebuild_model: 재구축 여부와 데이터 제한 수를 포함하는 모델

    Returns:
        구축 결과 메시지
    """
    try:
        await vector_service.process_and_build_vector_store(
            limit=rebuild_model.limit,
            force_rebuild=rebuild_model.force_rebuild
        )

        return {
            "message": "벡터 스토어 구축 완료",
            "force_rebuild": rebuild_model.force_rebuild,
            "limit": rebuild_model.limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 스토어 구축 중 오류 발생: {str(e)}")