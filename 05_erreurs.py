# Ce fichier gère les erreurs courantes lors de l'utilisation de l'API, telles que les erreurs d'authentification, les limites de taux et les problèmes de connexion. Il implémente une fonction de retry avec un backoff exponentiel pour gérer les erreurs temporaires.



from litellm import completion
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError
)
from dotenv import load_dotenv
import time

load_dotenv()

def ask_with_retry(question: str, max_retries: int = 3) -> str:
    """Pose une question avec gestion des erreurs et retries."""

    for attempt in range(max_retries):
        try:
            response = completion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": question}],
                timeout=30
            )
            return response.choices[0].message.content

        except AuthenticationError:
            raise Exception("❌ Clé API invalide. Vérifiez OPENAI_API_KEY.")

        except RateLimitError:
            wait = 2 ** attempt  # Backoff exponentiel
            print(f"⏳ Rate limit atteint, attente {wait}s...")
            time.sleep(wait)

        except APIConnectionError:
            print(f"🔌 Connexion échouée, tentative {attempt + 1}/{max_retries}")
            time.sleep(1)

    raise Exception("❌ Échec après plusieurs tentatives")

# Test
try:
    answer = ask_with_retry("Qu'est-ce que Kubernetes ?")
    print(answer)
except Exception as e:
    print(e)