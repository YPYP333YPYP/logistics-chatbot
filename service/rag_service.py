from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_, text

from db.database import async_get_db

# 필요한 라이브러리 임포트
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# 데이터베이스 관련 임포트
from sqlalchemy.sql.expression import case

# 벡터 스토어 및 임베딩 관련 임포트
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# LLM 관련 임포트
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 검색 및 RAG 구성 요소
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from model.logistics_shipment import LogisticsShipment


class RAGService:
    def __init__(
            self,
            model_name: str = "jhgan/ko-sroberta-multitask",
            llm_model_name: str = "beomi/KoAlpaca-Polyglot-5.8B",
            vector_store_path: str = "./data/vector_store",
            db: AsyncSession = Depends(async_get_db)
    ):
        """
        물류 배송 RAG 서비스 초기화 (오픈소스 모델 사용)

        Args:
            model_name: 임베딩 모델 이름 (HuggingFace)
            llm_model_name: LLM 모델 이름 (HuggingFace)
            vector_store_path: 벡터 스토어 경로
        """
        # 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"}
        )

        # 벡터 스토어 초기화
        self.vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings
        )

        # LLM 모델 초기화
        self._init_llm(llm_model_name)

        # 검색기 설정
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 압축 검색기를 사용하지 않고 기본 검색기만 사용 (리소스 효율성)
        self.qa_chain = self._create_qa_chain()

        self.db = db

        # 물류 온톨로지 정의 (기본 개념 및 관계)
        self.ontology = self._define_logistics_ontology()

    def _init_llm(self, model_name: str):
        """LLM 모델 초기화"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True if os.environ.get("USE_8BIT", "0") == "1" else False
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

    def _define_logistics_ontology(self) -> Dict[str, Any]:
        """
        물류 온톨로지 정의 (핵심 개념 및 관계)
        이를 통해 질의 의도에 따른 쿼리 매핑 가능
        """
        return {
            "concepts": {
                "Shipment": {"table": "logistics_shipments", "id_field": "lss_id"},
                "Port": {"related_fields": ["pol_nm", "pod_nm", "fd_nm"]},
                "Vessel": {"related_fields": ["pol_vessel_flight_nm", "pod_vessel_flight_nm"]},
                "Schedule": {"related_fields": ["pol_initial_etd", "pol_as_is_etd", "pol_atd",
                                                "pod_initial_eta", "pod_as_is_eta"]},
                "TransShipment": {"related_fields": ["ts_yn", "delivery_lane"]},
                "Status": {"related_fields": ["current_status", "current_location"]},
                "LeadTime": {"related_fields": ["lt_day_pol_atd_pod_initial_eta",
                                                "lt_day_pol_atd_pod_as_is_eta"]}
            },
            "relations": {
                "has_departed": {
                    "concept": "Shipment",
                    "condition": "pol_atd IS NOT NULL"
                },
                "has_arrived": {
                    "concept": "Shipment",
                    "condition": "current_status = 'ARRIVED' OR current_status = 'DELIVERED'"
                },
                "is_delayed": {
                    "concept": "Shipment",
                    "condition": "pol_as_is_etd > pol_initial_etd OR pod_as_is_eta > pod_initial_eta"
                },
                "has_transhipment": {
                    "concept": "Shipment",
                    "condition": "ts_yn = True"
                }
            },
            "intent_mapping": {
                "departure_status": ["has_departed", "Shipment", "Port"],
                "arrival_status": ["has_arrived", "Shipment", "Port"],
                "schedule_change": ["is_delayed", "Shipment", "Schedule"],
                "current_location": ["Status", "Shipment"],
                "lead_time": ["LeadTime", "Shipment"],
                "transhipment": ["has_transhipment", "Shipment", "TransShipment"]
            }
        }

    def _create_qa_chain(self):
        """질의응답 체인 생성"""
        prompt_template = """
        당신은 물류 배송 전문가입니다. 아래 제공된 컨텍스트에 기반하여 질문에 정확하게 대답하세요.
        답변을 모를 경우, "해당 정보를 찾을 수 없습니다"라고 대답하세요.

        컨텍스트: {context}

        질문: {question}

        답변:
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    async def query(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 검색

        Args:
            question: 사용자 질문

        Returns:
            답변과 관련 문서를 포함한 딕셔너리
        """
        # 의도 분류 (간단한 키워드 기반 매핑)
        intent = self._classify_intent(question)

        # 의도에 따른 관련 문서 검색 (하이브리드 검색)
        additional_docs = []
        if intent:
            db_results = await self._get_db_results_by_intent(intent, question)
            if db_results:
                additional_docs = [Document(page_content=str(r), metadata={"source": "database"}) for r in db_results]

        # 벡터 검색 결과와 DB 검색 결과 결합
        search_docs = self.retriever.get_relevant_documents(question)
        combined_docs = search_docs + additional_docs

        # 컨텍스트 생성
        context = "\n\n".join([doc.page_content for doc in combined_docs])

        # LLM으로 답변 생성
        answer = self.llm.predict(f"질문: {question}\n\n컨텍스트: {context}\n\n답변:")

        return {
            "answer": answer,
            "source_documents": combined_docs,
            "detected_intent": intent
        }

    def _classify_intent(self, question: str) -> Optional[str]:
        """질문 의도 분류"""
        # 핵심 키워드 추출 및 의도 매핑 (간단한 구현)
        intent_keywords = {
            "departure_status": ["출항", "출발", "ETD", "ATD", "출하"],
            "arrival_status": ["입항", "도착", "ETA", "ATA", "입하"],
            "schedule_change": ["변경", "지연", "차이", "이월", "지연율", "정시율"],
            "current_location": ["현재", "위치", "현황", "어디", "어디쯤"],
            "lead_time": ["리드타임", "소요", "기간", "며칠", "일수"],
            "transhipment": ["T/S", "환적", "트랜스", "다이렉트"]
        }

        for intent, keywords in intent_keywords.items():
            if any(keyword in question for keyword in keywords):
                return intent

        return None

    async def _get_db_results_by_intent(self, intent: str, question: str) -> List[Dict]:
        """의도에 따른 DB 쿼리 실행"""
        try:
            if intent == "departure_status":
                return await self._query_departure_status(question)
            elif intent == "arrival_status":
                return await self._query_arrival_status(question)
            elif intent == "schedule_change":
                return await self._query_schedule_changes(question)
            elif intent == "current_location":
                return await self._query_current_location(question)
            elif intent == "lead_time":
                return await self._query_lead_time(question)
            elif intent == "transhipment":
                return await self._query_transhipment(question)
            else:
                return []
        except Exception as e:
            print(f"DB 쿼리 오류: {e}")
            return []

    async def _query_departure_status(self, question: str) -> List[Dict]:
        """출항 상태 관련 쿼리"""
        # 특정 ID 검출 (예: H B/L, M B/L 등)
        hbl_id = self._extract_id_from_question(question, "hbl_no")
        mbl_id = self._extract_id_from_question(question, "mbl_no")
        lss_id = self._extract_id_from_question(question, "lss_id")

        stmt = select(
            LogisticsShipment.lss_id,
            LogisticsShipment.pol_nm,
            LogisticsShipment.pol_initial_etd,
            LogisticsShipment.pol_as_is_etd,
            LogisticsShipment.pol_atd,
            LogisticsShipment.current_status,
            LogisticsShipment.pol_vessel_flight_nm
        )

        if hbl_id:
            stmt = stmt.where(LogisticsShipment.hbl_no == hbl_id)
        elif mbl_id:
            stmt = stmt.where(LogisticsShipment.mbl_no == mbl_id)
        elif lss_id:
            stmt = stmt.where(LogisticsShipment.lss_id == lss_id)
        else:
            # 특정 ID가 없는 경우 최근 10개 레코드만 반환
            stmt = stmt.order_by(desc(LogisticsShipment.pol_atd)).limit(10)

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def _query_arrival_status(self, question: str) -> List[Dict]:
        """입항 상태 관련 쿼리"""
        # 유사한 패턴으로 구현
        hbl_id = self._extract_id_from_question(question, "hbl_no")

        stmt = select(
            LogisticsShipment.lss_id,
            LogisticsShipment.pod_nm,
            LogisticsShipment.pod_initial_eta,
            LogisticsShipment.pod_as_is_eta,
            LogisticsShipment.current_status,
            LogisticsShipment.pod_vessel_flight_nm
        )

        if hbl_id:
            stmt = stmt.where(LogisticsShipment.hbl_no == hbl_id)
        else:
            stmt = stmt.order_by(desc(LogisticsShipment.pod_as_is_eta)).limit(10)

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def _query_schedule_changes(self, question: str) -> List[Dict]:
        """스케줄 변경 관련 쿼리"""
        # 기본 필드 선택
        stmt = select(
            LogisticsShipment.lss_id,
            LogisticsShipment.pol_nm,
            LogisticsShipment.pod_nm,
            LogisticsShipment.pol_initial_etd,
            LogisticsShipment.pol_as_is_etd,
            LogisticsShipment.pol_atd,
            LogisticsShipment.pod_initial_eta,
            LogisticsShipment.pod_as_is_eta,
            # 출항 지연 일수 계산
            (func.julianday(LogisticsShipment.pol_atd) -
             func.julianday(LogisticsShipment.pol_initial_etd)).label('departure_delay_days'),
            # 예상 변경 일수 계산
            (func.julianday(LogisticsShipment.pod_as_is_eta) -
             func.julianday(LogisticsShipment.pod_initial_eta)).label('eta_change_days')
        ).where(
            or_(
                LogisticsShipment.pol_atd > LogisticsShipment.pol_initial_etd,
                LogisticsShipment.pod_as_is_eta > LogisticsShipment.pod_initial_eta
            )
        ).order_by(desc('departure_delay_days')).limit(10)

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def _query_current_location(self, question: str) -> List[Dict]:
        """현재 위치 관련 쿼리"""
        hbl_id = self._extract_id_from_question(question, "hbl_no")
        lss_id = self._extract_id_from_question(question, "lss_id")

        stmt = select(
            LogisticsShipment.lss_id,
            LogisticsShipment.hbl_no,
            LogisticsShipment.current_status,
            LogisticsShipment.current_location,
            LogisticsShipment.pol_nm,
            LogisticsShipment.pod_nm,
            LogisticsShipment.from_site_name,
            LogisticsShipment.to_site_name
        )

        if hbl_id:
            stmt = stmt.where(LogisticsShipment.hbl_no == hbl_id)
        elif lss_id:
            stmt = stmt.where(LogisticsShipment.lss_id == lss_id)
        else:
            stmt = stmt.limit(10)

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def _query_lead_time(self, question: str) -> List[Dict]:
        """리드타임 관련 쿼리"""
        # 출발지/도착지 추출
        from_port = self._extract_location_from_question(question, "from")
        to_port = self._extract_location_from_question(question, "to")

        stmt = select(
            LogisticsShipment.lss_id,
            LogisticsShipment.from_site_name,
            LogisticsShipment.to_site_name,
            LogisticsShipment.pol_nm,
            LogisticsShipment.pod_nm,
            LogisticsShipment.lt_day_pol_atd_pod_initial_eta,
            LogisticsShipment.lt_day_pol_atd_pod_as_is_eta,
            (func.julianday(LogisticsShipment.pod_as_is_eta) -
             func.julianday(LogisticsShipment.pol_atd)).label('actual_leadtime_days')
        )

        if from_port:
            stmt = stmt.where(
                or_(
                    LogisticsShipment.from_site_name.like(f"%{from_port}%"),
                    LogisticsShipment.pol_nm.like(f"%{from_port}%")
                )
            )

        if to_port:
            stmt = stmt.where(
                or_(
                    LogisticsShipment.to_site_name.like(f"%{to_port}%"),
                    LogisticsShipment.pod_nm.like(f"%{to_port}%")
                )
            )

        stmt = stmt.order_by(desc('actual_leadtime_days')).limit(10)

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def _query_transhipment(self, question: str) -> List[Dict]:
        """환적 관련 쿼리"""
        # 환적 여부 필터링
        stmt = select(
            LogisticsShipment.lss_id,
            LogisticsShipment.hbl_no,
            LogisticsShipment.ts_yn,
            LogisticsShipment.delivery_lane,
            LogisticsShipment.from_site_name,
            LogisticsShipment.to_site_name,
            LogisticsShipment.pol_nm,
            LogisticsShipment.pod_nm,
            LogisticsShipment.current_status,
            LogisticsShipment.current_location
        ).where(LogisticsShipment.ts_yn == True)

        # 특정 T/S 포트 검색
        ts_port = None
        for word in question.split():
            # 여기서는 간단한 구현으로 대체
            if len(word) > 3 and word not in ["포트", "항구", "출발", "도착"]:
                # 실제로는 지명 사전 등을 활용하여 더 정교하게 구현 필요
                ts_port = word
                break

        if ts_port:
            # 현재 모델에서는 T/S 포트를 직접 저장하는 필드가 없어
            # delivery_lane에 포함된 항구 정보를 검색
            stmt = stmt.where(LogisticsShipment.delivery_lane.like(f"%{ts_port}%"))

        stmt = stmt.limit(10)

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    def _extract_id_from_question(self, question: str, id_type: str) -> Optional[str]:
        """질문에서 식별자 추출 (간단한 구현)"""
        # 실제로는 정규표현식 등을 활용하여 더 정교하게 구현 필요
        words = question.split()

        # 특정 ID 패턴 매칭
        if id_type == "hbl_no" and "H B/L" in question:
            idx = question.find("H B/L")
            if idx >= 0 and idx + 10 < len(question):
                candidate = question[idx + 6:idx + 16].strip()
                if candidate and not candidate.isalpha():
                    return candidate

        elif id_type == "mbl_no" and "M B/L" in question:
            idx = question.find("M B/L")
            if idx >= 0 and idx + 10 < len(question):
                candidate = question[idx + 6:idx + 16].strip()
                if candidate and not candidate.isalpha():
                    return candidate

        elif id_type == "lss_id" and "LSS" in question:
            idx = question.find("LSS")
            if idx >= 0 and idx + 15 < len(question):
                candidate = question[idx + 3:idx + 13].strip()
                if candidate and not candidate.isalpha():
                    return candidate

        return None

    def _extract_location_from_question(self, question: str, direction: str) -> Optional[str]:
        """질문에서 위치 정보 추출 (간단한 구현)"""
        if direction == "from":
            keywords = ["에서", "출발", "출항", "시작"]
            for keyword in keywords:
                idx = question.find(keyword)
                if idx > 5:  # 적어도 몇 글자 앞에 위치 정보가 있어야 함
                    candidate = question[max(0, idx - 15):idx].strip()
                    if candidate:
                        return candidate.split()[-1]  # 마지막 단어만 추출
        elif direction == "to":
            keywords = ["으로", "로", "도착", "입항", "행"]
            for keyword in keywords:
                idx = question.find(keyword)
                if idx > 5:
                    candidate = question[max(0, idx - 15):idx].strip()
                    if candidate:
                        return candidate.split()[-1]

        return None

    async def get_shipment_by_id(self, lss_id: str) -> Optional[Dict[str, Any]]:
        """배송 ID로 배송 정보 검색"""
        stmt = select(LogisticsShipment).where(LogisticsShipment.lss_id == lss_id)
        result = await self.db.execute(stmt)
        shipment = result.scalars().first()

        if shipment:
            return {c.name: getattr(shipment, c.name) for c in shipment.__table__.columns}
        return None

    async def get_shipments_by_status(self, status: str, limit: int = 10) -> List[Dict[str, Any]]:
        """상태별 배송 정보 검색"""
        stmt = select(LogisticsShipment).where(
            LogisticsShipment.current_status == status
        ).limit(limit)

        result = await self.db.execute(stmt)
        shipments = result.scalars().all()

        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in shipments]

    async def get_delayed_shipments(self, days_threshold: int = 1) -> List[Dict[str, Any]]:
        """지연된 배송 정보 검색"""
        stmt = select(LogisticsShipment).where(
            or_(
                # 출항 지연
                and_(
                    LogisticsShipment.pol_atd != None,
                    LogisticsShipment.pol_initial_etd != None,
                    func.julianday(LogisticsShipment.pol_atd) -
                    func.julianday(LogisticsShipment.pol_initial_etd) > days_threshold
                ),
                # 도착 예상 지연
                and_(
                    LogisticsShipment.pod_initial_eta != None,
                    LogisticsShipment.pod_as_is_eta != None,
                    func.julianday(LogisticsShipment.pod_as_is_eta) -
                    func.julianday(LogisticsShipment.pod_initial_eta) > days_threshold
                )
            )
        )

        result = await self.db.execute(stmt)
        shipments = result.scalars().all()

        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in shipments]

    async def get_transhipment_shipments(self) -> List[Dict[str, Any]]:
        """T/S 배송 정보 검색"""
        stmt = select(LogisticsShipment).where(LogisticsShipment.ts_yn == True)

        result = await self.db.execute(stmt)
        shipments = result.scalars().all()

        return [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in shipments]

    async def calculate_port_delay_statistics(self, port_name: str, months: int = 3) -> Dict[str, Any]:
        """특정 항구의 지연 통계 계산"""
        # 기준 날짜 계산 (현재 날짜로부터 X개월 전)
        cutoff_date = datetime.now() - timedelta(days=30 * months)

        # 출항 지연 통계
        departure_stmt = select(
            func.avg(
                func.julianday(LogisticsShipment.pol_atd) -
                func.julianday(LogisticsShipment.pol_initial_etd)
            ).label('avg_departure_delay'),
            func.max(
                func.julianday(LogisticsShipment.pol_atd) -
                func.julianday(LogisticsShipment.pol_initial_etd)
            ).label('max_departure_delay'),
            func.min(
                func.julianday(LogisticsShipment.pol_atd) -
                func.julianday(LogisticsShipment.pol_initial_etd)
            ).label('min_departure_delay')
        ).where(
            LogisticsShipment.pol_nm.like(f"%{port_name}%"),
            LogisticsShipment.pol_atd >= cutoff_date
        )

        departure_result = await self.db.execute(departure_stmt)
        departure_stats = departure_result.first()

        # 도착 지연 통계
        arrival_stmt = select(
            func.avg(
                func.julianday(LogisticsShipment.pod_as_is_eta) -
                func.julianday(LogisticsShipment.pod_initial_eta)
            ).label('avg_arrival_delay'),
            func.max(
                func.julianday(LogisticsShipment.pod_as_is_eta) -
                func.julianday(LogisticsShipment.pod_initial_eta)
            ).label('max_arrival_delay'),
            func.min(
                func.julianday(LogisticsShipment.pod_as_is_eta) -
                func.julianday(LogisticsShipment.pod_initial_eta)
            ).label('min_arrival_delay')
        ).where(
            LogisticsShipment.pod_nm.like(f"%{port_name}%"),
            LogisticsShipment.pod_as_is_eta >= cutoff_date
        )

        arrival_result = await self.db.execute(arrival_stmt)
        arrival_stats = arrival_result.first()

        return {
            "port_name": port_name,
            "period_months": months,
            "departure_stats": dict(departure_stats._mapping) if departure_stats else {},
            "arrival_stats": dict(arrival_stats._mapping) if arrival_stats else {}
        }

    async def calculate_carrier_performance(
            self, carrier_name: str = None, months: int = 3
    ) -> List[Dict[str, Any]]:
        """선사별 성능 통계 계산"""
        cutoff_date = datetime.now() - timedelta(days=30 * months)

        stmt = select(
            LogisticsShipment.carrier_nm,
            func.count(LogisticsShipment.id).label('shipment_count'),
            func.avg(
                func.julianday(LogisticsShipment.pol_atd) -
                func.julianday(LogisticsShipment.pol_initial_etd)
            ).label('avg_departure_delay'),
            func.avg(
                case(
                    (
                        LogisticsShipment.pol_atd <= LogisticsShipment.pol_initial_etd,
                        1
                    ),
                    else_=0
                )
            ).label('departure_ontime_rate')
        ).where(
            LogisticsShipment.pol_atd >= cutoff_date
        ).group_by(LogisticsShipment.carrier_nm)

        if carrier_name:
            stmt = stmt.where(LogisticsShipment.carrier_nm.like(f"%{carrier_name}%"))

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def calculate_route_performance(
            self, from_port: str = None, to_port: str = None, months: int = 3
    ) -> List[Dict[str, Any]]:
        """경로별 성능 통계 계산"""
        cutoff_date = datetime.now() - timedelta(days=30 * months)

        stmt = select(
            LogisticsShipment.pol_nm,
            LogisticsShipment.pod_nm,
            LogisticsShipment.ts_yn,
            func.count(LogisticsShipment.id).label('shipment_count'),
            func.avg(
                func.julianday(LogisticsShipment.pod_as_is_eta) -
                func.julianday(LogisticsShipment.pol_atd)
            ).label('avg_leadtime'),
            func.avg(
                func.julianday(LogisticsShipment.pod_as_is_eta) -
                func.julianday(LogisticsShipment.pod_initial_eta)
            ).label('avg_arrival_delay'),
            func.avg(
                case(
                    (
                        LogisticsShipment.pod_as_is_eta <= LogisticsShipment.pod_initial_eta,
                        1
                    ),
                    else_=0
                )
            ).label('arrival_ontime_rate')
        ).where(
            LogisticsShipment.pol_atd >= cutoff_date
        ).group_by(
            LogisticsShipment.pol_nm,
            LogisticsShipment.pod_nm,
            LogisticsShipment.ts_yn
        )

        if from_port:
            stmt = stmt.where(LogisticsShipment.pol_nm.like(f"%{from_port}%"))

        if to_port:
            stmt = stmt.where(LogisticsShipment.pod_nm.like(f"%{to_port}%"))

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    async def weekly_performance_trend(
            self, from_port: str = None, to_port: str = None, weeks: int = 12
    ) -> List[Dict[str, Any]]:
        """주간 성능 추이 계산"""
        cutoff_date = datetime.now() - timedelta(days=7 * weeks)

        stmt = select(
            LogisticsShipment.pol_etd_week,
            func.count(LogisticsShipment.id).label('shipment_count'),
            func.avg(
                func.julianday(LogisticsShipment.pod_as_is_eta) -
                func.julianday(LogisticsShipment.pol_atd)
            ).label('avg_leadtime'),
            func.avg(
                case(
                    (
                        and_(
                            LogisticsShipment.pol_atd <= LogisticsShipment.pol_initial_etd,
                            LogisticsShipment.pod_as_is_eta <= LogisticsShipment.pod_initial_eta
                        ),
                        1
                    ),
                    else_=0
                )
            ).label('overall_ontime_rate')
        ).where(
            LogisticsShipment.pol_atd >= cutoff_date
        ).group_by(
            LogisticsShipment.pol_etd_week
        ).order_by(LogisticsShipment.pol_etd_week)

        if from_port:
            stmt = stmt.where(LogisticsShipment.pol_nm.like(f"%{from_port}%"))

        if to_port:
            stmt = stmt.where(LogisticsShipment.pod_nm.like(f"%{to_port}%"))

        result = await self.db.execute(stmt)
        rows = result.all()
        return [dict(row._mapping) for row in rows]

    def _shipment_to_document(self, shipment: Dict[str, Any]) -> Document:
        """배송 정보를 문서로 변환"""
        # 텍스트 표현 생성
        content = f"""
        물류 배송 ID: {shipment.get('lss_id', '')}
        이름: {shipment.get('lss_name', '')}
        현재 상태: {shipment.get('current_status', '')}
        현재 위치: {shipment.get('current_location', '')}
        환적 여부: {'예' if shipment.get('ts_yn') else '아니오'}

        참조 번호:
        - SR No: {shipment.get('sr_no', '')}
        - H B/L No: {shipment.get('hbl_no', '')}
        - M B/L No: {shipment.get('mbl_no', '')}
        - ULD No: {shipment.get('uld_no', '')}

        서비스 정보:
        - 서비스명: {shipment.get('service_nm', '')}
        - 서비스 유형: {shipment.get('d_s_t', '')}
        - 서비스 기간: {shipment.get('service_term', '')}
        - 인코텀즈: {shipment.get('incoterms', '')}

        출발지/도착지:
        - 출발지: {shipment.get('from_site_name', '')} (ID: {shipment.get('from_site_id', '')})
        - 도착지: {shipment.get('to_site_name', '')} (ID: {shipment.get('to_site_id', '')})

        화주/수하인:
        - 화주: {shipment.get('shipper_name', '')} (ID: {shipment.get('shipper_id', '')})
        - 수하인: {shipment.get('consignee_nm', '')} (ID: {shipment.get('consignee_id', '')})

        공장 정보:
        - 출발 공장: {shipment.get('from_plant_nm', '')} (코드: {shipment.get('from_plant_cd', '')})
        - 도착 공장: {shipment.get('to_plant_nm', '')} (코드: {shipment.get('to_plant_cd', '')})

        경로 정보:
        - 배송 노선: {shipment.get('delivery_lane', '')}
        - 출발항: {shipment.get('pol_nm', '')} ({shipment.get('pol_nation', '')}, ID: {shipment.get('pol_loc_id', '')})
        - 도착항: {shipment.get('pod_nm', '')} ({shipment.get('pod_nation', '')}, ID: {shipment.get('pod_loc_id', '')})
        - 최종 도착지: {shipment.get('fd_nm', '')} ({shipment.get('fd_nation_nm', '')}, ID: {shipment.get('fd_loc_id', '')})

        운송사 정보:
        - LSP: {shipment.get('lsp_nm', '')} (ID: {shipment.get('lsp_id', '')})
        - 선사: {shipment.get('carrier_nm', '')} (코드: {shipment.get('carrier_code', '')})

        출항 정보 (POL):
        - 선박/항공편: {shipment.get('pol_vessel_flight_nm', '')} (번호: {shipment.get('pol_vessel_flight_no', '')})
        - 최초 예상 출항일: {shipment.get('pol_initial_etd', '')}
        - 현재 예상 출항일: {shipment.get('pol_as_is_etd', '')}
        - 실제 출항일: {shipment.get('pol_atd', '')}
        - 출항 주차: {shipment.get('pol_etd_week', '')}
        - 실제 출항 주차: {shipment.get('pol_atd_week', '')}
        - 출항 정시 상태: {shipment.get('pol_ontime_status', '')}
        - 출항 지연 기간: {shipment.get('pol_aging_period', '')} 일

        도착 정보 (POD):
        - 선박/항공편: {shipment.get('pod_vessel_flight_nm', '')} (번호: {shipment.get('pod_vessel_flight_no', '')})
        - 최초 예상 도착일: {shipment.get('pod_initial_eta', '')}
        - 현재 예상 도착일: {shipment.get('pod_as_is_eta', '')}

        리드타임:
        - 실제 출항-최초 예상 도착 리드타임: {shipment.get('lt_day_pol_atd_pod_initial_eta', '')} 일
        - 실제 출항-현재 예상 도착 리드타임: {shipment.get('lt_day_pol_atd_pod_as_is_eta', '')} 일
        """

        return Document(
            page_content=content,
            metadata={
                "id": shipment.get('lss_id', ''),
                "source": "logistics_shipments",
                "current_status": shipment.get('current_status', ''),
                "from_site": shipment.get('from_site_name', ''),
                "to_site": shipment.get('to_site_name', '')
            }
        )

    async def update_vector_store(self, shipment_id: str = None):
        """벡터 스토어 업데이트"""
        if shipment_id:
            # 특정 배송 정보만 업데이트
            shipment = await self.get_shipment_by_id(shipment_id)
            if not shipment:
                return {"status": "error", "message": f"배송 ID {shipment_id}를 찾을 수 없습니다."}

            documents = [self._shipment_to_document(shipment)]
        else:
            # 전체 배송 정보 업데이트
            stmt = select(LogisticsShipment)
            result = await self.db.execute(stmt)
            shipments = result.scalars().all()

            documents = [self._shipment_to_document(
                {c.name: getattr(s, c.name) for c in s.__table__.columns}
            ) for s in shipments]

        # 벡터 스토어에 문서 추가
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

        return {
            "status": "success",
            "message": f"{len(documents)}개 문서가 벡터 스토어에 업데이트되었습니다."
        }