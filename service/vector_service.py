import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, inspect, String, Integer, Float, Boolean, DateTime
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

        # 백터 스토어 저장 경로 초기화
        self.vector_store_path = vector_store_path

        # 비동기 세션 초기화
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
        query = select(LogisticsShipment)
        if limit:
            query = query.limit(limit)
        result = await self.session.execute(query)
        shipments_orm = result.scalars().all()

        # 모델 컬럼 정보 미리 가져오기
        mapper = inspect(LogisticsShipment)
        column_types = {column.key: column.type for column in mapper.columns}

        shipments = []
        for shipment in shipments_orm:
            shipment_dict = {}

            # 모든 속성 순회
            for key, value in shipment.__dict__.items():
                # 내부 속성 제외
                if key.startswith('_'):
                    continue

                # None 값 처리
                if value is None:
                    # 컬럼 타입 확인
                    if key in column_types:
                        column_type = column_types[key]

                        # 컬럼 타입에 따른 기본값 설정
                        if isinstance(column_type, String):
                            shipment_dict[key] = "없음"
                        elif isinstance(column_type, Integer):
                            shipment_dict[key] = 0
                        elif isinstance(column_type, Float):
                            shipment_dict[key] = 0.0
                        elif isinstance(column_type, Boolean):
                            shipment_dict[key] = False
                        elif isinstance(column_type, DateTime):
                            shipment_dict[key] = "1970-01-01T00:00:00"
                        else:
                            # 기타 타입은 빈 문자열로 설정
                            shipment_dict[key] = ""
                    else:
                        # 컬럼 정보가 없는 경우 빈 문자열로 설정
                        shipment_dict[key] = ""
                else:
                    # None이 아닌 값 처리
                    if isinstance(value, datetime):
                        # 날짜 객체는 ISO 형식 문자열로 변환
                        shipment_dict[key] = value.isoformat()
                    else:
                        # 기타 타입은 그대로 저장
                        shipment_dict[key] = value

            # 임베딩 생성 전 데이터 유효성 확인
            for key, value in shipment_dict.items():
                if value == "":
                    # 핵심 필드의 빈 문자열 처리
                    if key in ['lss_id', 'lss_name', 'current_status']:
                        shipment_dict[key] = f"unknown_{key}"

            # 결과 리스트에 추가
            shipments.append(shipment_dict)

        # 벡터 임베딩 생성 전 최종 확인
        for shipment in shipments:
            # 필수 필드가 비어있는지 확인
            if 'lss_id' not in shipment or not shipment['lss_id']:
                shipment['lss_id'] = f"default_id_{shipment.get('id', 'unknown')}"

            # 벡터 임베딩에 사용될 텍스트 필드 전처리
            text_fields = ['lss_name', 'current_status', 'current_location', 'service_nm']
            for field in text_fields:
                if field in shipment and (not shipment[field] or shipment[field] == ""):
                    shipment[field] = "정보_없음"

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


    def search_with_filter(self, query: str, filter_dict: Dict[str, Any], k: int = 5) -> List[Document]:
        """
        필터를 적용하여 문서 검색

        Args:
            query (str): 검색 쿼리
            filter_dict (Dict): 필터 조건 (메타데이터 기반)
            k (int): 반환할 문서 수

        Returns:
            List[Document]: 필터링된 유사한 문서 목록
        """
        if self.vector_store is None:
            print("벡터 스토어가 초기화되지 않았습니다. 먼저 process_and_build_vector_store()를 실행하세요.")
            return []

        # Chroma 필터 형식으로 변환 (수정된 부분)
        if len(filter_dict) > 1:
            # 여러 조건이 있는 경우 $and 연산자 사용
            filter_conditions = []
            for key, value in filter_dict.items():
                filter_conditions.append({key: {"$eq": value}})
            chroma_filter = {"$and": filter_conditions}
        elif len(filter_dict) == 1:
            # 단일 조건인 경우 직접 사용
            key, value = next(iter(filter_dict.items()))
            chroma_filter = {key: {"$eq": value}}
        else:
            # 필터가 없는 경우
            chroma_filter = None

        # 벡터 스토어에서 필터링된 유사 문서 검색
        docs = self.vector_store.similarity_search(
            query,
            k=k,
            filter=chroma_filter
        )
        return docs

    async def export_structured_data(self, output_path: str = "./data/structured_data"):
        """
        RDB 데이터를 구조화된 형태로 내보내기 (비동기)

        Args:
            output_path (str): 출력 파일 저장 경로
        """
        os.makedirs(output_path, exist_ok=True)

        # 화물 데이터 내보내기
        shipments = await self.fetch_shipment_data()
        with open(os.path.join(output_path, "shipments.json"), "w", encoding="utf-8") as f:
            json.dump(shipments, f, ensure_ascii=False, indent=2)

        # 데이터프레임으로 변환하여 CSV로도 저장
        shipments_df = pd.DataFrame(shipments)
        shipments_df.to_csv(os.path.join(output_path, "shipments.csv"), index=False, encoding="utf-8")

        print(f"구조화된 데이터가 {output_path}에 저장되었습니다.")

    # TODO: 컨테이너, 스케줄, 항구, 선사 등의 데이터 처리 메서드 구현 예정


# main.py 테스트 코드
async def vector_service_lss_example():
    """LSS ID 기반 물류 검색 예제 함수"""

    # 비동기 세션 생성
    async for session in async_get_db():
        # 벡터 서비스 초기화
        vector_service = VectorService(session=session)

        # 벡터 스토어가 없으면 구축
        if vector_service.vector_store is None:
            print("벡터 스토어 구축 중...")
            await vector_service.process_and_build_vector_store()

        # 예제 1: LSS ID로 특정 화물 검색
        lss_id = "T523"
        query1 = f"{lss_id} 화물의 현재 위치와 상태"

        print(f"\n===== {lss_id} 화물 검색 결과 =====")
        docs1 = vector_service.search_similar_documents(query1, k=2)
        for i, doc in enumerate(docs1, 1):
            print(f"\n결과 {i}:")
            print(f"내용: {doc.page_content[:200]}...")
            print(f"메타데이터: {doc.metadata}")

        # 예제 2: LSS ID와 함께 필터링 (특정 화물의 환적 정보)
        filter_query1 = {
            "lss_id": lss_id,
            "ts_yn": True
        }

        print(f"\n===== {lss_id} 환적 화물 필터 검색 결과 =====")
        try:
            # 여러 조건의 필터 검색 - 수정된 방식 적용
            filter_conditions = []
            for key, value in filter_query1.items():
                filter_conditions.append({key: {"$eq": value}})

            chroma_filter = {"$and": filter_conditions}

            filtered_docs1 = vector_service.vector_store.similarity_search(
                "환적 정보",
                k=2,
                filter=chroma_filter
            )

            for i, doc in enumerate(filtered_docs1, 1):
                print(f"\n결과 {i}:")
                print(f"내용: {doc.page_content[:200]}...")
                print(f"메타데이터: {doc.metadata}")
        except Exception as e:
            print(f"필터 검색 오류: {e}")
            print("단일 필터 검색으로 시도합니다...")
            # 단일 필터로 시도
            docs = vector_service.search_with_filter(
                "환적 정보",
                {"lss_id": lss_id},
                k=2
            )
            for i, doc in enumerate(docs, 1):
                print(f"\n결과 {i}:")
                print(f"내용: {doc.page_content[:200]}...")
                print(f"메타데이터: {doc.metadata}")

        # 예제 3: 두 개의 LSS ID 비교 검색
        other_lss_id = "514A"
        query2 = f"{lss_id}와 {other_lss_id} 화물의 운송 경로 비교"

        print(f"\n===== {lss_id}와 {other_lss_id} 비교 검색 결과 =====")
        docs2 = vector_service.search_similar_documents(query2, k=3)
        for i, doc in enumerate(docs2, 1):
            print(f"\n결과 {i}:")
            print(f"내용: {doc.page_content[:200]}...")
            print(f"메타데이터: {doc.metadata}")

        # 예제 4: 특정 LSS ID 화물의 출항 지연 여부 확인
        query3 = f"{lss_id} 화물의 출항 예정일과 실제 출항일 비교"

        print(f"\n===== {lss_id} 출항 지연 여부 검색 결과 =====")
        docs3 = vector_service.search_similar_documents(query3, k=2)
        for i, doc in enumerate(docs3, 1):
            print(f"\n결과 {i}:")
            print(f"내용: {doc.page_content[:200]}...")
            print(f"메타데이터: {doc.metadata}")

            # 출항 지연 여부 분석
            if "pol_initial_etd" in doc.page_content and "pol_atd" in doc.page_content:
                print("\n[출항 지연 분석]")
                print("최초 출항 예정일과 실제 출항일 정보가 있습니다.")
                print("지연 여부를 확인하려면 날짜를 비교하세요.")

        # 예제 5: 특정 LSS ID 화물과 동일 선박에 실린 다른 화물 검색
        # 먼저 해당 LSS ID의 선박 정보 찾기
        vessel_query = f"{lss_id} 화물의 선박 정보"
        vessel_docs = vector_service.search_similar_documents(vessel_query, k=1)

        if vessel_docs:
            # 선박 정보 추출 시도
            vessel_info = None
            doc_content = vessel_docs[0].page_content
            vessel_lines = [line for line in doc_content.split('\n') if "선박/항공편 정보" in line]

            if vessel_lines:
                vessel_info = vessel_lines[0].split(":")[-1].strip()
                print(f"\n===== {lss_id} 화물과 동일 선박({vessel_info})에 실린 화물 검색 =====")

                if vessel_info and vessel_info != "":
                    same_vessel_query = f"{vessel_info} 선박에 실린 화물 목록"
                    same_vessel_docs = vector_service.search_similar_documents(same_vessel_query, k=3)

                    for i, doc in enumerate(same_vessel_docs, 1):
                        print(f"\n결과 {i}:")
                        print(f"내용: {doc.page_content[:200]}...")
                        print(f"메타데이터: {doc.metadata}")
                else:
                    print(f"선박 정보를 찾을 수 없습니다.")
            else:
                print(f"선박 정보를 찾을 수 없습니다.")

        # 한 번만 실행
        break