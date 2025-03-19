import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server

from model import carrier, logistics_shipment
from db.database import init_db, close_db, async_get_db
from api.routes import router
from service.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    try:
        await init_db()  # ✅ DB 초기화 실행
        os.environ["USE_CUDA"] = "1" if os.path.exists("/dev/nvidia0") else "0"

    except Exception as e:
        yield
        return
    yield
    try:
        await close_db()  # ✅ DB 종료 실행
    except Exception as e:
        print(e)


app = FastAPI(lifespan=lifespan)


# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


async def test_rag_service():
    async for db in async_get_db():
        try:
            print("RAG 서비스 초기화 중...")
            # 사용 가능한 모델이 없는 경우를 위한 기본값 설정
            # 실제 환경에 맞게 모델 경로를 조정하세요
            rag_service = RAGService(
                model_name="jhgan/ko-sroberta-multitask",  # 임베딩 모델
                llm_model_name="beomi/KoAlpaca-Polyglot-5.8B",  # LLM 모델
                vector_store_path="./data/vector_store",
                db=db
            )

            print("RAG 서비스 초기화 완료!")

            # 샘플 질문 목록
            test_questions = [
                "부산에서 상해로 가는 배송의 현재 상태는 어떤가요?",
                "H B/L 번호 12345678로 배송 상태를 알려주세요.",
                "지난주에 지연된 배송은 몇 건입니까?",
                "롱비치 항구의 평균 지연 시간은 얼마입니까?",
                "환적이 필요한 배송의 평균 리드타임은 얼마입니까?"
            ]

            # 각 질문에 대해 RAG 서비스 쿼리 실행
            for i, question in enumerate(test_questions):
                print(f"\n질문 {i + 1}: {question}")
                try:
                    result = rag_service.query(question)
                    print(f"답변: {result['answer']}")
                    print(f"검색된 의도: {result['detected_intent']}")
                    print(f"관련 문서 수: {len(result['source_documents'])}")
                except Exception as e:
                    print(f"쿼리 처리 중 오류 발생: {e}")

            # 다른 메서드 테스트 (필요에 따라)
            print("\n지연된 배송 정보 검색:")
            try:
                delayed_shipments = rag_service.get_delayed_shipments(days_threshold=1)
                print(f"지연된 배송 수: {len(delayed_shipments)}")
                if delayed_shipments:
                    sample = delayed_shipments[0]
                    print(f"샘플 배송 ID: {sample.get('lss_id')}")
                    print(f"출발항: {sample.get('pol_nm')}")
                    print(f"도착항: {sample.get('pod_nm')}")
            except Exception as e:
                print(f"지연 배송 검색 중 오류 발생: {e}")

        except Exception as e:
            print(f"RAG 서비스 초기화 중 오류 발생: {e}")
        finally:
            # 데이터베이스 세션 정리
            await db.close()


async def run_server():
    config = Config(app=app, host="127.0.0.1", port=8000, log_level="debug")
    server = Server(config=config)
    await test_rag_service()
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_server())