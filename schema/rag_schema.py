from typing import Optional, List, Dict, Any

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    detected_intent: Optional[str] = None
    source_documents: Optional[List[Dict[str, Any]]] = None


class ShipmentRequest(BaseModel):
    lss_id: str


class ShipmentListResponse(BaseModel):
    shipments: List[Dict[str, Any]]
    count: int


class StatisticsResponse(BaseModel):
    statistics: Dict[str, Any]


class UpdateResponse(BaseModel):
    status: str
    message: str