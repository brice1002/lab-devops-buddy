"""
Router basique pour gérer plusieurs providers avec des retries et des fallback.
Ce fichier montre comment utiliser la classe Router de litellm pour configurer 
plusieurs providers d'AI et gérer les retries et les fallbacks.
Il utilise la fonction completion pour obtenir une reponse du modele de maniere 
asynchrone.
Il utilise un pattern pour trouver les fichiers DevOps courants dans un projet.
Il affiche un resume a la fin avec le nombre de fichiers analyses, le temps total 
et le gain par rapport a une analyse sequentielle.
Il cree aussi des fichiers de test pour simuler un projet DevOps avec 
un Dockerfile, un docker-compose et un fichier GitLab CI.
Il utilise un system prompt pour orienter les reponses du modele vers une 
analyse concise et pertinente des fichiers DevOps.
Il affiche les resultats de chaque analyse avec le nom du fichier, le temps 
d'analyse et le contenu de l'analyse.
Il gère les erreurs courantes comme les erreurs de connexion, les limites de 
taux et les erreurs d'authentification en utilisant des retries avec un backoff 
exponentiel.
"""


from litellm import Router
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration des providers
model_list = [
    {
        "model_name": "devops-buddy",  # Alias unique pour l'app. model_name est utilisé pour identifier le provider dans les appels à router.completion(model="devops-buddy", ...)
        "litellm_params": {
            "model": "openai/gpt-4.1-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "order": 1  # Provider principal
        }
    },
    {
        "model_name": "devops-buddy",
        "litellm_params": {
            "model": "anthropic/claude-3-5-haiku-latest",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "order": 2  # Premier fallback
        }
    },
    # {
    #     "model_name": "devops-buddy",
    #     "litellm_params": {
    #         "model": "gemini/gemini-2.0-flash",
    #         "api_key": os.getenv("GOOGLE_API_KEY"),
    #         "order": 3  # Dernier recours
    #     }
    # }
]

# Router avec retries et cooldowns
router = Router(
    model_list=model_list,
    num_retries=2,           # Réessaie 2x avant fallback
    retry_after=1,           # 1s entre retries
    timeout=30,              # Max 30s par requête
    allowed_fails=3,         # Échecs avant cooldown
    cooldown_time=60,        # Désactive 60s si trop d'échecs
    enable_pre_call_checks=True
)

# Test
response = router.completion(
    model="devops-buddy",
    messages=[
        {"role": "system", "content": "Tu es DevOps Buddy, assistant DevOps expert."},
        {"role": "user", "content": "Comment vérifier les logs d'un pod K8s ?"}
    ]
)

print(f"✅ Provider utilisé: {response.model}")
print(f"📝 Réponse: {response.choices[0].message.content[:200]}...")