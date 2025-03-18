from typing import Optional

from pydantic import BaseModel


class RebuildModel(BaseModel):
    force_rebuild: Optional[bool] = False
    limit: Optional[int] = None