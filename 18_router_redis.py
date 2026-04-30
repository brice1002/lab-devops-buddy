


from litellm import Router
from dotenv import load_dotenv
import os

load_dotenv()

model_list = [
    {
        "model_name": "devops-buddy",
        "litellm_params": {
            "model": "openai/gpt-4.1-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
]

router = Router(
    model_list=model_list,
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=int(os.getenv("REDIS_PORT", 6379)),
    redis_password=os.getenv("REDIS_PASSWORD", None),
    cache_responses=True
)