from typing import Optional

from fastapi import APIRouter, Depends, Body, HTTPException, Query

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


@router.post("/update")
async def update_vector_store(
        vector_service: VectorService = Depends(VectorService)
):
    """
    벡터 스토어 증분 업데이트 API

    Returns:
        업데이트 결과 메시지
    """
    try:
        if vector_service.vector_store is None:
            await vector_service.process_and_build_vector_store()
            message = "벡터 스토어가 존재하지 않아 새로 구축합니다."
        else:
            await vector_service.update_vector_store()
            message = "벡터 스토어 증분 업데이트 완료"

        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 스토어 업데이트 중 오류 발생: {str(e)}")


@router.get("/export")
async def export_data(
        output_path: Optional[str] = Query("./data/structured_data"),
        vector_service: VectorService = Depends()
):
    """
    구조화된 데이터 내보내기 API

    Args:
        output_path: 출력 파일 저장 경로

    Returns:
        내보내기 결과 메시지
    """
    try:
        await vector_service.export_structured_data(output_path)
        return {
            "message": "구조화된 데이터 내보내기 완료",
            "output_path": output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 내보내기 중 오류 발생: {str(e)}")