from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from schema.rag_schema import QueryResponse, QueryRequest, UpdateResponse, StatisticsResponse, ShipmentListResponse
from service.rag_service import RAGService

router = APIRouter(
    prefix="/api/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}}
)

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, service: RAGService = Depends(RAGService)):
    """
    물류 배송에 관한 자연어 질의에 응답합니다.
    """
    try:
        result = service.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류 발생: {str(e)}")

@router.get("/shipment/{lss_id}", response_model=Dict[str, Any])
async def get_shipment(lss_id: str, service: RAGService = Depends(RAGService)):
    """
    특정 배송 ID의 상세 정보를 조회합니다.
    """
    shipment = service.get_shipment_by_id(lss_id)
    if not shipment:
        raise HTTPException(status_code=404, detail=f"배송 ID {lss_id}를 찾을 수 없습니다.")
    return shipment

@router.get("/shipments/status/{status}", response_model=ShipmentListResponse)
async def get_shipments_by_status(
    status: str,
    limit: int = Query(10, ge=1, le=100),
    service: RAGService = Depends(RAGService)
):
    """
    특정 상태의 배송 목록을 조회합니다.
    """
    shipments = service.get_shipments_by_status(status, limit)
    return ShipmentListResponse(shipments=shipments, count=len(shipments))

@router.get("/shipments/delayed", response_model=ShipmentListResponse)
async def get_delayed_shipments(
    days_threshold: int = Query(1, ge=0),
    service: RAGService = Depends(RAGService)
):
    """
    지연된 배송 목록을 조회합니다.
    """
    shipments = service.get_delayed_shipments(days_threshold)
    return ShipmentListResponse(shipments=shipments, count=len(shipments))

@router.get("/shipments/transhipment", response_model=ShipmentListResponse)
async def get_transhipment_shipments(service: RAGService = Depends(RAGService)):
    """
    환적 배송 목록을 조회합니다.
    """
    shipments = service.get_transhipment_shipments()
    return ShipmentListResponse(shipments=shipments, count=len(shipments))

@router.get("/statistics/port/{port_name}", response_model=StatisticsResponse)
async def get_port_statistics(
    port_name: str,
    months: int = Query(3, ge=1, le=24),
    service: RAGService = Depends(RAGService)
):
    """
    특정 항구의 지연 통계를 조회합니다.
    """
    statistics = service.calculate_port_delay_statistics(port_name, months)
    return StatisticsResponse(statistics=statistics)

@router.get("/statistics/carrier", response_model=Dict[str, Any])
async def get_carrier_performance(
    carrier_name: Optional[str] = None,
    months: int = Query(3, ge=1, le=24),
    service: RAGService = Depends(RAGService)
):
    """
    선사별 성능 통계를 조회합니다.
    """
    statistics = service.calculate_carrier_performance(carrier_name, months)
    return {"carriers": statistics, "count": len(statistics)}

@router.get("/statistics/route", response_model=Dict[str, Any])
async def get_route_performance(
    from_port: Optional[str] = None,
    to_port: Optional[str] = None,
    months: int = Query(3, ge=1, le=24),
    service: RAGService = Depends(RAGService)
):
    """
    경로별 성능 통계를 조회합니다.
    """
    statistics = service.calculate_route_performance(from_port, to_port, months)
    return {"routes": statistics, "count": len(statistics)}

@router.get("/statistics/weekly-trend", response_model=Dict[str, Any])
async def get_weekly_trend(
    from_port: Optional[str] = None,
    to_port: Optional[str] = None,
    weeks: int = Query(12, ge=4, le=52),
    service: RAGService = Depends(RAGService)
):
    """
    주간 성능 추이를 조회합니다.
    """
    trend_data = service.weekly_performance_trend(from_port, to_port, weeks)
    return {"trends": trend_data, "count": len(trend_data)}

@router.post("/vector-store/update", response_model=UpdateResponse)
async def update_vector_store(
    shipment_id: Optional[str] = None,
    service: RAGService = Depends(RAGService)
):
    """
    벡터 스토어를 업데이트합니다.
    """
    result = service.update_vector_store(shipment_id)
    return result