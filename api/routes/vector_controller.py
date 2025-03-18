from typing import Optional

from fastapi import APIRouter, Depends, Body, HTTPException, Query

from schema.vector_schema import RebuildModel, QueryModel
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
        vector_service: VectorService = Depends(VectorService)
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


@router.get("/status")
async def check_vector_store_status(
        vector_service: VectorService = Depends(VectorService)
):
    """
    벡터 스토어 상태 확인 API

    Returns:
        벡터 스토어 상태 정보
    """
    try:
        is_initialized = vector_service.vector_store is not None

        status_info = {
            "initialized": is_initialized,
        }

        if is_initialized:
            document_count = vector_service.vector_store._collection.count()
            status_info["document_count"] = document_count
            status_info["path"] = vector_service.vector_store._persist_directory

        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 스토어 상태 확인 중 오류 발생: {str(e)}")


@router.post("/search")
async def search_documents(
        query_model: QueryModel,
        vector_service: VectorService = Depends(VectorService)
):
    """
    쿼리와 유사한 문서 검색 API

    Args:
        query_model: 검색 쿼리, 결과 수, 필터 조건을 포함하는 모델

    Returns:
        검색 결과 목록
    """
    try:
        if vector_service.vector_store is None:
            # 벡터 스토어가 초기화되지 않은 경우 자동 구축
            await vector_service.process_and_build_vector_store()

        if query_model.filter_dict:
            # 필터가 있는 경우 필터 검색 실행
            docs = vector_service.search_with_filter(
                query_model.query,
                query_model.filter_dict,
                query_model.k
            )
        else:
            # 일반 검색 실행
            docs = vector_service.search_similar_documents(
                query_model.query,
                query_model.k
            )

        # Document 객체를 dict로 변환하여 반환
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")
