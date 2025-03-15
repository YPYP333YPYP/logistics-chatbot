from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import carrier, shipping
from db.database import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    try:
        await init_db()  # ✅ DB 초기화 실행

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
