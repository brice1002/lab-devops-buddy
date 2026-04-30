"""
Ce fichier montre comment limiter le nombre d'appels API simultanés en utilisant un sémaphore asyncio. 
Cela permet de respecter les limites de taux de l'API tout en traitant plusieurs tâches en parallèle.

Le code utilise la fonction acompletion pour obtenir une réponse du modèle de maniere asynchrone. 
Il utilise asyncio.Semaphore pour limiter le nombre de tâches en cours en utilisant un sémaphore. 
Il utilise un pattern pour trouver les fichiers DevOps courants dans un projet.
Il affiche un résumé à la fin avec le nombre de fichiers analysés, le temps total et le gain par rapport à une analyse séquentielle.
Il crée aussi des fichiers de test pour simuler un projet DevOps avec un Dockerfile, un docker-compose et un fichier GitLab CI.
Il utilise un system prompt pour orienter les réponses du modèle vers une analyse concise et pertinente des fichiers DevOps.
Il affiche les résultats de chaque analyse avec le nom du fichier, le temps d'analyse et le contenu de l'analyse.
"""


import asyncio
from pathlib import Path
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

# Maximum 5 appels API simultanés
MAX_CONCURRENT = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT) # Crée un sémaphore pour limiter le nombre de tâches concurrentes à 5 en utilisant asyncio Semaphore


## Cette fonction analyse un fichier en utilisant le modèle, mais elle utilise un sémaphore pour limiter le nombre d'appels API simultanés. Si 5 appels sont déjà en cours, les autres attendront leur tour avant de s'exécuter. Cela permet de respecter les limites de taux de l'API tout en traitant plusieurs fichiers en parallèle.
async def analyze_with_limit(filepath: Path, system_prompt: str) -> dict:
    """Analyse un fichier avec contrôle du parallélisme.""" 

    async with semaphore:  # Attend son tour si 5 appels sont déjà en cours
        content = filepath.read_text()

        response = await acompletion(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyse: {filepath.name}\n\n{content}"}
            ]
        )

        return {
            "file": filepath.name,
            "result": response.choices[0].message.content
        }


async def analyze_many(files: list[Path]) -> list[dict]:
    """Analyse plusieurs fichiers avec limite de concurrence."""

    system_prompt = "Tu es un auditeur DevOps. Analyse brièvement ce fichier."

    # Lance toutes les tâches — le semaphore contrôle le débit
    tasks = [analyze_with_limit(f, system_prompt) for f in files]
    results = await asyncio.gather(*tasks)

    return results


async def main():
    # Simule 15 fichiers à analyser
    files = [Path(f"config_{i}.yml") for i in range(15)]
    for f in files:
        f.write_text(f"# Config {f.name}\nkey: value")

    print(f"🔍 Analyse de {len(files)} fichiers (max {MAX_CONCURRENT} simultanés)...")
    results = await analyze_many(files)
    print(f"✅ {len(results)} fichiers analysés")

    # Nettoyage
    for f in files:
        f.unlink()


asyncio.run(main())