# init_db.py
from config.db_config import Base, engine
from models.candle import Candle

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Done.")
