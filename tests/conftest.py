"""
Global pytest configuration.
- Automatically loads environment variables from .env.dev (or user-specified file).
- Silences SQLAlchemy logging to keep test output clean.
- Provides shared fixtures for DB setup or other resources if needed.
Example:
    pytest -v
    pytest -v --env=prod
    pytest -v --env=test
"""

import os
import logging
import pytest
from dotenv import load_dotenv


def pytest_addoption(parser):
    """
    Add a custom command-line option to select environment file.
    Example:
        pytest --env=prod
    """
    parser.addoption(
        "--env",
        action="store",
        default="dev",
        help="Select environment configuration: dev / test / prod",
    )


@pytest.fixture(scope="session", autouse=True)
def setup_environment(request):
    """
    Automatically load environment variables before all tests.
    Also silences SQLAlchemy logs globally.
    """
    # ---- Determine environment file ----
    env_name = request.config.getoption("--env")
    env_file = f".env.{env_name}"
    env_path = os.path.join(os.getcwd(), env_file)

    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"\nLoaded environment file: {env_file}")
    else:
        print(f"\nEnvironment file not found: {env_file}. Using system defaults.")

    # ---- Silence noisy loggers ----
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)

    # Optional: reduce other common test noise
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("faker.factory").setLevel(logging.ERROR)

    # ---- Confirm DB URL presence ----
    db_url = os.getenv("DATABASE_URL", None)
    if not db_url:
        print("DATABASE_URL not set. Tests may fail if DB connection is required.")
    else:
        print(f"Database URL detected (hidden): {db_url[:8]}********")

    yield  # (You could clean up resources here if needed)


@pytest.fixture(scope="session")
def test_env():
    """
    Provide environment metadata for tests.
    Example usage:
        def test_env_loaded(test_env):
            assert test_env["ENV_NAME"] == "dev"
    """
    return {
        "ENV_NAME": os.getenv("ENV_NAME", "dev"),
        "DATABASE_URL": os.getenv("DATABASE_URL", "not_set"),
    }
