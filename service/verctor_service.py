import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import Depends
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from db.database import async_get_db

from model.logistics_shipment import LogisticsShipment


class VectorService:
    """
    물류 데이터를 RDB에서 읽어와 벡터 스토어로 변환하는 서비스
    """

    def __init__(self,
                 session: AsyncSession = Depends(async_get_db),
                 embedding_model_name: str = "jhgan/ko-sroberta-multitask",
                 vector_store_path: str = "./data/vector_store"):
        """
        VectorService 초기화

        Args:
            session (AsyncSession): RDB에 대한 비동기 세션
            embedding_model_name (str): 임베딩에 사용할 모델 이름
            vector_store_path (str): 벡터 스토어 저장 경로
        """

        self.vector_store_path = vector_store_path
        self.session = session

        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # 벡터 스토어 경로 생성
        os.makedirs(vector_store_path, exist_ok=True)

        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # 기존 벡터 스토어 로드 또는 새로 생성
        if os.path.exists(os.path.join(vector_store_path, "chroma.sqlite3")):
            self.vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=self.embeddings
            )
            print(f"기존 벡터 스토어 로드 완료. 문서 수: {self.vector_store._collection.count()}")
        else:
            self.vector_store = None
            print("벡터 스토어가 존재하지 않습니다. 데이터 처리 후 새로 생성됩니다.")

    async def fetch_shipment_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        RDB에서 물류 화물 데이터 비동기 조회

        Args:
            limit (int, optional): 조회할 최대 레코드 수

        Returns:
            List[Dict[str, Any]]: 화물 데이터 목록
        """
        # SQLAlchemy 2.0 스타일로 쿼리 구성
        query = select(LogisticsShipment)

        # 제한이 있으면 적용
        if limit:
            query = query.limit(limit)

        # 쿼리 실행
        result = await self.session.execute(query)
        shipments_orm = result.scalars().all()

        # ORM 객체를 딕셔너리로 변환
        shipments = []
        for shipment in shipments_orm:
            # __dict__를 통해 객체 속성을 딕셔너리로 변환
            shipment_dict = {k: v for k, v in shipment.__dict__.items() if not k.startswith('_')}

            # 날짜/시간 객체 문자열로 변환
            for key, value in shipment_dict.items():
                if isinstance(value, datetime):
                    shipment_dict[key] = value.isoformat()

            shipments.append(shipment_dict)

        return shipments

    def convert_shipment_to_document(self, shipment: Dict[str, Any]) -> Document:
        """
        화물 데이터를 Document 객체로 변환

        Args:
            shipment (Dict[str, Any]): 화물 데이터

        Returns:
            Document: 변환된 Document 객체
        """
        # 화물 정보를 문자열로 구성
        content = f"""
      화물 ID: {shipment.get('lss_id', '')}
      화물 이름: {shipment.get('lss_name', '')}
      현재 상태: {shipment.get('current_status', '')}
      현재 위치: {shipment.get('current_location', '')}
      환적 여부: {'예' if shipment.get('ts_yn') else '아니오'}
      B/L 번호: HBL-{shipment.get('hbl_no', '')}, MBL-{shipment.get('mbl_no', '')}
      SR 번호: {shipment.get('sr_no', '')}
      서비스: {shipment.get('service_nm', '')}
      출발지: {shipment.get('from_site_name', '')} ({shipment.get('from_site_id', '')})
      도착지: {shipment.get('to_site_name', '')} ({shipment.get('to_site_id', '')})
      화주: {shipment.get('shipper_name', '')}
      수하인: {shipment.get('consignee_nm', '')}
      물류 경로: {shipment.get('delivery_lane', '')}
      선적항(POL): {shipment.get('pol_nm', '')} ({shipment.get('pol_nation', '')})
      도착항(POD): {shipment.get('pod_nm', '')} ({shipment.get('pod_nation', '')})
      최종 목적지: {shipment.get('fd_nm', '')} ({shipment.get('fd_nation_nm', '')})
      운송사: {shipment.get('carrier_nm', '')} ({shipment.get('carrier_code', '')})
      선박/항공편 정보: {shipment.get('pol_vessel_flight_nm', '')} {shipment.get('pol_vessel_flight_no', '')}
      출항 예정일(최초): {shipment.get('pol_initial_etd', '')}
      출항 예정일(현재): {shipment.get('pol_as_is_etd', '')}
      출항일: {shipment.get('pol_atd', '')}
      도착 예정일(최초): {shipment.get('pod_initial_eta', '')}
      도착 예정일(현재): {shipment.get('pod_as_is_eta', '')}
      예상 리드타임(일): {shipment.get('lt_day_pol_atd_pod_as_is_eta', '')}
      POL 정시 상태: {shipment.get('pol_ontime_status', '')}
      """

        raw_metadata = {
            "shipment_id": shipment.get('id'),
            "lss_id": shipment.get('lss_id'),
            "current_status": shipment.get('current_status'),
            "pol": shipment.get('pol_nm'),
            "pod": shipment.get('pod_nm'),
            "carrier": shipment.get('carrier_nm'),
            "ts_yn": shipment.get('ts_yn'),
            "doc_type": "shipment"
        }

        # None 값 필터링
        metadata = {k: (v if v is not None else "") for k, v in raw_metadata.items()}

        return Document(page_content=content, metadata=metadata)

