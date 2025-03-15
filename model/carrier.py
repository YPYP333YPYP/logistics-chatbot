from sqlalchemy import Column, Integer, String

from db.database import Base


class Carrier(Base):
    __tablename__ = "carriers"

    id = Column(Integer, primary_key=True, index=True)
    carrier_code = Column(String(10), nullable=False, index=True)
    carrier_name = Column(String(100), nullable=False)