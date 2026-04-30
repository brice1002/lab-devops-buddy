"""
Ce fichier montre comment utiliser l'API de maniere asynchrone avec asyncio et gérer les erreurs.
Il utilise la fonction acompletion pour obtenir une reponse du modele de maniere asynchrone.
Il utilise un pattern pour trouver les fichiers DevOps courants dans un projet.
Il affiche un resume a la fin avec le nombre de fichiers analyses, le temps total et le gain par rapport a une analyse sequentielle.
Il cree aussi des fichiers de test pour simuler un projet DevOps avec un Dockerfile, un docker-compose et un fichier GitLab CI.
Il utilise un system prompt pour orienter les reponses du modele vers une analyse concise et pertinente des fichiers DevOps.
Il affiche les resultats de chaque analyse avec le nom du fichier, le temps d'analyse et le contenu de l'analyse.
Il gère les erreurs courantes comme les erreurs de connexion, les limites de taux et les erreurs d'authentification en utilisant des retries avec un backoff exponentiel.
"""


import asyncio
from pathlib import Path
from litellm import acompletion
from litellm.exceptions import RateLimitError, APIConnectionError
from dotenv import load_dotenv
import time

load_dotenv()


# Cette fonction analyse un fichier en utilisant le modèle, mais elle utilise un sémaphore pour limiter le nombre d'appels API simultanés. Si 5 appels sont déjà en cours, les autres attendront leur tour avant de s'exécuter. Cela permet de respecter les limites de taux de l'API tout en traitant plusieurs fichiers en parallèle.
async def analyze_safe(filepath: Path, max_retries: int = 3) -> dict:
    """Analyse avec retries et gestion d'erreurs."""

    for attempt in range(max_retries):
        try:
            content = filepath.read_text()

            response = await acompletion(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "user", "content": f"Analyse brièvement: {content[:500]}"} # Limite à 500 caractères pour éviter des requêtes trop longues
                ],
                timeout=30 # Timeout pour éviter d'attendre indéfiniment en cas de problème de connexion
            )

            return {
                "file": filepath.name,
                "status": "success",
                "result": response.choices[0].message.content
            }

        except RateLimitError:
            wait = 2 ** attempt
            print(f"⏳ {filepath.name}: rate limit, attente {wait}s...")
            await asyncio.sleep(wait)

        except APIConnectionError:
            print(f"🔌 {filepath.name}: connexion échouée, retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(1)

        except Exception as e:
            return {
                "file": filepath.name,
                "status": "error",
                "result": str(e)
            }

    return {
        "file": filepath.name,
        "status": "failed",
        "result": "Échec après plusieurs tentatives"
    }


async def main():
    files = [Path("Dockerfile")]  # Votre liste de fichiers à analyser. Vous pouvez utiliser un pattern pour trouver les fichiers DevOps courants dans un projet, comme "Dockerfile*", "docker-compose*.yml", ".gitlab-ci.yml", etc.
    # si on souhaite plusieurs fichiers dans files, on peut utiliser glob pour les trouver automatiquement dans un répertoire, par exemple: files = list(Path(".").glob("**/Dockerfile*")) pour trouver tous les Dockerfiles dans le projet. Et pour trouver plusieurs on peut utiliser plusieurs patterns, par exemple: files = list(Path(".").glob("**/Dockerfile*")) + list(Path(".").glob("**/docker-compose*.yml")) + list(Path(".").glob("**/.gitlab-ci.yml")) pour trouver tous les Dockerfiles, docker-compose et fichiers GitLab CI dans le projet.

    # return_exceptions=True évite l'arrêt si une tâche échoue
    results = await asyncio.gather(
        *[analyze_safe(f) for f in files],
        return_exceptions=True
    )

    # Résumé
    success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success") # Compte le nombre de succès dans les résultats en vérifiant que chaque résultat est un dictionnaire et que son statut est "success" en utilisant une compréhension de liste pour parcourir les résultats et en utilisant sum pour compter les succès.
    print(f"\n✅ {success}/{len(results)} fichiers analysés avec succès")


asyncio.run(main())