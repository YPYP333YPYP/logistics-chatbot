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


