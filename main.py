import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server

from model import carrier, logistics_shipment
from db.database import init_db, close_db
from api.routes import router

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

app.include_router(router)


async def run_server():
    config = Config(app=app, host="127.0.0.1", port=8000, log_level="debug")
    server = Server(config=config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_server())