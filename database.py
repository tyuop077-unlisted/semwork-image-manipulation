import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    data_source_type = Column(String, index=True)

    data_source_info = Column(JSON)

    augmentation_details = Column(JSON, nullable=True)

    result_method1 = Column(Integer, nullable=True)
    params_method1 = Column(JSON, nullable=True) # -- 1

    result_method2 = Column(Integer, nullable=True)
    params_method2 = Column(JSON, nullable=True) # -- 2

    result_method3 = Column(Integer, nullable=True)
    params_method3 = Column(JSON, nullable=True) # -- 3

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
