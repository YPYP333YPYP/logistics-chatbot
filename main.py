import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime

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


async def test_rag_service_with_samples():
    """
    테스트 함수로 RAG 서비스를 초기화하고 샘플 질문들에 대한 응답을 JSON 파일로 저장합니다.
    """
    results = []

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

            # 샘플 질문 목록 (기존 질문 + 새 질문)
            test_questions = [
                # 기존 질문
                "부산에서 상해로 가는 배송의 현재 상태는 어떤가요?",
                "H B/L 번호 12345678로 배송 상태를 알려주세요.",
                "지난주에 지연된 배송은 몇 건입니까?",
                "롱비치 항구의 평균 지연 시간은 얼마입니까?",
                "환적이 필요한 배송의 평균 리드타임은 얼마입니까?",

                # 추가된 질문들
                "출항완료 했나요?",
                "출항일이 변경되었나요?",
                "입항 했나요?",
                "입항일이 변경되었나요?",
                "최종 도착지에 도착했나요?",
                "상세 스케줄을 알려주세요",
                "스케줄의 변동여부를 상세하게 알려주세요",
                "화물의 다음 기항지는 어디인가요?",
                "현재 화물의 배송 현황을 알려주세요",
                "현재 화물의 위치를 알려주세요",
                "다이렉트 스케줄인가요?",
                "T/S 스케줄만 알려주세요",
                "화물이 T/S 포트에 입항되었나요?",
                "화물이 T/S 포트에서 출항했나요?",
                "T/S 스케줄 변동 여부 알려주세요 (최초 vs 현재 스케줄)",
                "T/S 출항까지 몇 시간 대기해야하나요? (T/S ETA-ETD)",
                "OOO포트에서의 T/S 실제 평균 소요 시간 알려주세요",
                "어디서 T/S 예정인가요?",
                "몇번 째 T/S 인가요?",
                "몇 번 T/S 예정인가요?",
                "T/S 상세 스케줄 알려주세요 (POD, ETD, ETA)",
                "최초 출항일 대비 실제 출항일 몇일 차이나나요?",
                "최초 입항일 대비 실제 입항일 몇일 차이나나요?",
                "최근 3개월 내 OOO포트에서의 T/S가 3일 이상 지연된 적 있었나요?",
                "총 몇일 소요되나요?",
                "OOO포트에서 도착지(CY/Door-Door)까지 몇일 소요되나요?",
                "지금으로부터 도착지까지 몇일 남았나요?",
                "최초 리드타임과 실제 리드타임을 알려주세요",
                "OOO-OOO구간에 대해 어디에서 몇일 지연이 발생했는지 알려주세요",
                "최초 대비 몇일 변동되었는지 트랙킹 포인트별로 알려주세요"
            ]

            # BL 번호 예시 (실제 환경에 맞게 조정)
            sample_bl_number = "ABCD12345678"
            sample_origin = "부산"
            sample_destination = "로테르담"
            sample_ts_port = "싱가포르"

            # 각 질문에 대해 RAG 서비스 쿼리 실행
            for question in test_questions:
                print(f"\n질문: {question}")

                # BL 번호, 포트 이름 등 템플릿 변수 대체
                processed_question = question.replace("OOO포트", sample_origin).replace("H B/L 번호 12345678",
                                                                                      f"H B/L 번호 {sample_bl_number}")
                processed_question = processed_question.replace("OOO-OOO", f"{sample_origin}-{sample_destination}")

                try:
                    # RAG 서비스 쿼리 실행
                    result = await rag_service.query(processed_question)

                    # 결과 저장
                    qa_pair = {
                        "question": processed_question,
                        "answer": result['answer'],
                        "detected_intent": result['detected_intent'],
                        "source_documents_count": len(result['source_documents']),
                        "timestamp": datetime.now().isoformat()
                    }

                    results.append(qa_pair)
                    print(f"답변: {result['answer']}")

                except Exception as e:
                    print(f"쿼리 처리 중 오류 발생: {e}")

                    # 오류가 발생해도 결과에 기록
                    qa_pair = {
                        "question": processed_question,
                        "answer": f"오류 발생: {str(e)}",
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(qa_pair)

            # 결과를 JSON 파일로 저장
            output_filename = f"rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, ensure_ascii=False, indent=2)

            print(f"\n테스트 결과가 {output_filename}에 저장되었습니다.")

        except Exception as e:
            print(f"RAG 서비스 초기화 중 오류 발생: {e}")
        finally:
            # 데이터베이스 세션 정리
            await db.close()

    return results


async def run_server():
    config = Config(app=app, host="127.0.0.1", port=8000, log_level="debug")
    server = Server(config=config)
    await test_rag_service_with_samples()
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_server())