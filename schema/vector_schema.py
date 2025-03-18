from typing import Optional, Dict, Any

from pydantic import BaseModel


class RebuildModel(BaseModel):
    force_rebuild: Optional[bool] = False
    limit: Optional[int] = None


class QueryModel(BaseModel):
    query: str
    k: Optional[int] = 5
    filter_dict: Optional[Dict[str, Any]] = None