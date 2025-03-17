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

    async def process_and_build_vector_store(self, force_rebuild: bool = False):
        """
        RDB 데이터를 처리하고 벡터 스토어 구축 (비동기)

        Args:
            force_rebuild (bool): 기존 벡터 스토어가 있어도 강제로 재구축할지 여부
        """
        # 이미 벡터 스토어가 있고 강제 재구축이 아니면 스킵
        if self.vector_store is not None and not force_rebuild:
            print("벡터 스토어가 이미 존재합니다. force_rebuild=True로 설정하여 재구축할 수 있습니다.")
            return

        print("데이터 수집 및 벡터 스토어 구축 시작...")

        # 화물 데이터 수집 및 문서 변환
        print("화물 데이터 수집 중...")
        shipments = await self.fetch_shipment_data()
        print(f"총 {len(shipments)}개의 화물 데이터 수집 완료")

        shipment_docs = []
        for shipment in shipments:
            doc = self.convert_shipment_to_document(shipment)
            shipment_docs.append(doc)

        # 문서 분할
        print("문서 분할 중...")
        chunks = self.text_splitter.split_documents(shipment_docs)
        print(f"총 {len(chunks)}개의 청크로 분할 완료")

        # 벡터 스토어 구축
        print("벡터 스토어 구축 중...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        self.vector_store.persist()
        print("벡터 스토어 구축 및 저장 완료")

    async def update_vector_store(self):
        """
        벡터 스토어 증분 업데이트 (비동기)
        - 새로운 화물 데이터만 추가
        """
        if self.vector_store is None:
            print("벡터 스토어가 초기화되지 않았습니다. 먼저 process_and_build_vector_store()를 실행하세요.")
            return

        print("벡터 스토어 증분 업데이트 시작...")

        # 기존 문서의 shipment_id 수집
        existing_ids = set()
        for metadata in self.vector_store._collection.get()["metadatas"]:
            if "shipment_id" in metadata:
                existing_ids.add(metadata["shipment_id"])

        # 모든 화물 데이터 조회
        all_shipments = await self.fetch_shipment_data()

        # 새로운 화물만 필터링
        new_shipments = [s for s in all_shipments if s["id"] not in existing_ids]
        print(f"총 {len(new_shipments)}개의 새로운 화물 데이터 발견")

        if not new_shipments:
            print("업데이트할 새로운 데이터가 없습니다.")
            return

        # 문서 변환 및 추가
        new_docs = []
        for shipment in new_shipments:
            doc = self.convert_shipment_to_document(shipment)
            new_docs.append(doc)

        # 문서 분할
        chunks = self.text_splitter.split_documents(new_docs)

        # 벡터 스토어에 추가
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        print(f"{len(chunks)}개의 새로운 청크가 벡터 스토어에 추가되었습니다.")

    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        쿼리와 유사한 문서 검색

        Args:
            query (str): 검색 쿼리
            k (int): 반환할 문서 수

        Returns:
            List[Document]: 유사한 문서 목록
        """
        if self.vector_store is None:
            print("벡터 스토어가 초기화되지 않았습니다. 먼저 process_and_build_vector_store()를 실행하세요.")
            return []

        # 벡터 스토어에서 유사 문서 검색
        docs = self.vector_store.similarity_search(query, k=k)
        return docs


