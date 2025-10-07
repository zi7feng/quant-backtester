import os
from dotenv import load_dotenv

# default env "dev"
env = os.getenv("ENV", "dev")

if env == "prod":
    env_file = ".env.prod"
else:
    env_file = ".env.dev"

load_dotenv(env_file)

print(f"Loaded environment: {env} ({env_file})")