from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import async_get_db


class RagService:

    def __init__(self, session:AsyncSession = Depends(async_get_db)):
        self.session = session
