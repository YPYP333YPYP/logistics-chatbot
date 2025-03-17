from typing import AsyncGenerator

import pymysql
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

from core.config import RDBSettings

rdb_setting = RDBSettings()

# DATABASE_URL = f"mysql+aiomysql://{rdb_setting.RDB_USERNAME}:{rdb_setting.RDB_PASSWORD}@{rdb_setting.DB_HOST}:{rdb_setting.DB_PORT}/{rdb_setting.DB_DATABASE_NAME}"
DATABASE_URL = "mysql+aiomysql://root:root@localhost:3306/harin"

# todo - logger 사용

# 비동기 엔진 생성
async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
)
# 비동기 세션 생성
local_session = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


Base = declarative_base()


# 데이터베이스 생성 메서드
def create_database_if_not_exists():
    try:
        connection = pymysql.connect(
            host=rdb_setting.DB_HOST,
            user=rdb_setting.RDB_USERNAME,
            password=rdb_setting.RDB_PASSWORD
        )
        cursor = connection.cursor()

        cursor.execute("CREATE DATABASE IF NOT EXISTS harin")
        print("Database created or already exists.")

        cursor.close()
        connection.close()
    except pymysql.MySQLError as e:
        print(f"Error: {e}")


# 비동기 데이터베이스 세션 의존성
async def async_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with local_session() as session:
        try:
            yield session
        finally:
            await session.close()


# 데이터베이스 초기화 함수
async def init_db():
    try:
        print("start db")
        create_database_if_not_exists()
        # DB 연결 및 테이블 생성
        async with async_engine.begin() as conn:
            print("Creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            print("Tables created!")

    except Exception as e:
        print(f"Error during database initialization: {e}")


# 데이터베이스 종료 함수
async def close_db():
    print("close db")
    await async_engine.dispose()
