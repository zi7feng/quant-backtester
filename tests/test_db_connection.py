from config.db_config import SessionLocal
from sqlalchemy import text

def test_database_connection():
    """Ensure database connection is valid and accessible."""
    session = SessionLocal()
    try:
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1, "Database connection failed."
    finally:
        session.close()
