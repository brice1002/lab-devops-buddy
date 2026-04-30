"""
Monitoring et alerte :
Ce fichier montre comment créer un système de monitoring pour les performances 
et les coûts des modèles utilisés dans DevOps Buddy, avec des alertes en cas de 
dépassement de seuils définis.
Il définit une classe DevOpsLogger qui hérite de CustomLogger et implémente les 
méthodes log_success et log_failure pour enregistrer les succès et les échecs 
des appels aux modèles, ainsi que pour calculer des métriques agrégées telles 
que le taux de réussite, la latence moyenne et le taux de fallback.
Il utilise la classe Router de litellm pour envoyer des requêtes à un modèle de 
langage, en passant une instance de DevOpsLogger dans les callbacks pour 
enregistrer les métriques de chaque appel.
Il affiche les métriques agrégées à la fin des appels, montrant le nombre total 
de requêtes, le taux de réussite, la latence moyenne et le taux de fallback.
Il utilise des emojis pour indiquer visuellement les succès (✅) et les 
échecs (❌) dans les logs, ainsi que pour afficher les métriques agrégées (📊) .
"""


from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()


class DevOpsLogger(CustomLogger):
    """Logger custom pour DevOps Buddy."""

    def __init__(self):
        self.stats = {
            "success": 0,
            "failures": 0,
            "total_latency": 0,
            "fallbacks": 0
        }

    def log_success(self, kwargs, response, start_time, end_time):
        model = kwargs.get("model", "unknown")
        latency = end_time - start_time
        tokens = response.usage.total_tokens if response.usage else 0

        self.stats["success"] += 1
        self.stats["total_latency"] += latency

        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] "
              f"{model} | {latency:.2f}s | {tokens} tokens")

    def log_failure(self, kwargs, exception, start_time, end_time):
        model = kwargs.get("model", "unknown")

        self.stats["failures"] += 1

        print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] "
              f"{model} | {type(exception).__name__}: {exception}")

    def get_metrics(self) -> dict:
        """Retourne les métriques agrégées."""
        total = self.stats["success"] + self.stats["failures"]
        return {
            "total_requests": total,
            "success_rate": self.stats["success"] / max(total, 1) * 100,
            "avg_latency": self.stats["total_latency"] / max(self.stats["success"], 1),
            "fallback_rate": self.stats["fallbacks"] / max(total, 1) * 100
        }

class CustomRouter(Router):
    def __init__(self, model_list, callbacks):
        super().__init__(model_list)
        self.callbacks = callbacks

# Utilisation
logger = DevOpsLogger()

model_list = [
    {
        "model_name": "devops-buddy",
        "litellm_params": {
            "model": "openai/gpt-4.1-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
]

callbacks = [logger]
router = CustomRouter(
    model_list=model_list,
    callbacks=callbacks
)

# Test
for i in range(5):
    router.completion(
        model="devops-buddy",
        messages=[{"role": "user", "content": f"Question {i + 1}"}]
    )

print("\n📊 Métriques:")
for k, v in logger.get_metrics().items():
    print(f"  {k}: {v:.2f}")