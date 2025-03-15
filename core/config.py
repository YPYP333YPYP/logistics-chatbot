import os

import pymysql
from pydantic.v1 import BaseSettings


class RDBSettings(BaseSettings):
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: int = os.getenv("DB_PORT")
    DB_DATABASE_NAME: str = os.getenv("DB_DATABASE_NAME")
    RDB_USERNAME: str = os.getenv("RDB_USERNAME")
    RDB_PASSWORD: str = os.getenv("RDB_PASSWORD")

    class Config:
        env_file = ".env"