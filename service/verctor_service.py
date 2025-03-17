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

