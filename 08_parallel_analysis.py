"""# Ce fichier analyse plusieurs fichiers DevOps d'un projet en parallèle, 
en utilisant asyncio pour lancer plusieurs requêtes à l'API en même temps. 
Il mesure le temps total et compare avec le temps séquentiel pour montrer 
les gains de performance. 
Il affiche aussi le nombre de tokens utilisés pour chaque analyse. 
Le code est commenté pour mieux comprendre son fonctionnement. 
Il utilise la fonction acompletion pour obtenir une réponse du modèle de 
maniere asynchrone. 
Il utilise asyncio.gather pour lancer toutes les analyses en parallèle. 
Il utilise des patterns pour trouver les fichiers DevOps courants dans un projet.
Il affiche un résumé à la fin avec le nombre de fichiers analysés, 
le temps total et le gain par rapport à une analyse séquentielle. 
Il crée aussi des fichiers de test pour simuler un projet DevOps avec un 
Dockerfile, un docker-compose et un fichier GitLab CI. 
Il utilise un system prompt pour orienter les réponses du modèle vers une analyse concise et pertinente des fichiers DevOps. 
Il affiche les résultats de chaque analyse avec le nom du fichier, le temps d'analyse et le contenu de l'analyse.
"""


import asyncio
import time
from pathlib import Path
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """Tu es DevOps Buddy, un auditeur DevOps. Analyse ce fichier et donne :
1. Type de fichier
2. Problèmes trouvés (sécurité, performance)
3. Score /10

Sois concis (max 150 mots)."""


# Cette fonction analyse un fichier DevOps et retourne un rapport, en chronométrant le temps d'analyse pour mesurer les performances.
async def analyze_file(filepath: Path) -> dict:
    """Analyse un fichier et chronomètre."""
    start = time.time() # Chronomètre le début de l'analyse

    content = filepath.read_text()

    # Envoie une requête asynchrone à l'API pour analyser le fichier, en utilisant le system prompt pour orienter la réponse du modèle.
    response = await acompletion(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Fichier: {filepath.name}\n\n```\n{content}\n```"}
        ],
        # Limitation de 300 tokens pour éviter des réponses trop longues, car on veut une analyse concise.
        max_tokens=300
    )

    # Chronomètre la fin de l'analyse et calcule le temps écoulé.
    elapsed = time.time() - start

    return {
        "file": filepath.name,
        "time": elapsed, # Temps d'analyse pour ce fichier
        "analysis": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    }


# Cette fonction analyse tous les fichiers DevOps d'un projet en parallèle, en utilisant asyncio.gather pour lancer toutes les analyses en même temps.
async def analyze_project(directory: Path) -> list[dict]:
    """Analyse tous les fichiers DevOps d'un projet en parallèle."""

    # Fichiers DevOps à rechercher
    patterns = [
        "Dockerfile*",
        "docker-compose*.yml",
        "docker-compose*.yaml",
        ".gitlab-ci.yml",
        ".github/workflows/*.yml",
        "*.tf",
        "Jenkinsfile",
        "ansible/*.yml"
    ]

    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern)) # Recherche tous les fichiers correspondant aux patterns dans le répertoire donné

    if not files:
        print("⚠️ Aucun fichier DevOps trouvé")
        return []

    print(f"🔍 Analyse de {len(files)} fichiers en parallèle...\n")

    # Lancer TOUTES les analyses en parallèle
    results = await asyncio.gather(
        *[analyze_file(f) for f in files] # Analyse tous les fichiers en parallèle en créant une liste de tâches pour asyncio.gather et en appelant analyze_file pour chaque fichier en parallèle en utilisant asyncio.gather pour attendre que toutes les analyses soient terminées et récupérer les résultats dans une liste de dictionnaires contenant le nom du fichier, le temps d'analyse, l'analyse du modèle et le nombre de tokens utilisés.
    )

    return results


async def main():
    # Créer des fichiers de test pour simuler un projet DevOps avec un Dockerfile, un docker-compose et un fichier GitLab CI. Cela permet de tester la fonction d'analyse en parallèle avec des fichiers réels et de mesurer les performances.
    test_dir = Path(".") # Répertoire actuel pour les fichiers de test

    # Dockerfile de test avec des problèmes de sécurité (permissions 777) et de performance (pas d'optimisation des couches). Le modèle devrait identifier ces problèmes dans son analyse. Le contenu du Dockerfile est écrit de manière à inclure des problèmes courants pour que le modèle puisse les détecter et les analyser dans sa réponse. Le docker-compose et le fichier GitLab CI sont également conçus pour inclure des problèmes courants afin de tester la capacité du modèle à les identifier et à les analyser correctement.
    Path("Dockerfile").write_text("""FROM ubuntu:latest # Image de base Ubuntu latest avec les derniers paquets et mises à jour de sécurité dans le conteneur de base pour assurer une meilleure sécurité et compatibilité. 
RUN apt-get update && apt-get install -y python3 curl wget git
COPY . /app
RUN chmod 777 /app
CMD ["python3", "app.py"]
""")
    
#     # Meme chose pour windows avec des problèmes de security et de performance (pas d'optimisation des couches) pour tester la capacité du modèle à analyser différents types de fichiers et à identifier les problèmes spécifiques à chaque type de fichier. Le Dockerfile Windows inclut des commandes spécifiques à Windows et des problèmes courants pour que le modèle puisse les détecter et les analyser dans sa réponse. 
#     Path("Dockerfile.windows").write_text("""FROM mcr.microsoft.com/windows/server:ltsc2019
# RUN powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; `
#     [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
#     iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')); `
#     choco install git -y""")

    # docker-compose de test
    Path("docker-compose.yml").write_text("""version: '3.8' 
services:
  web:
    build: .
    ports:
      - "80:80"
    environment:
      - DEBUG=true
      - DB_PASSWORD=secret123
""")
    
#     # Meme chose pour windows
#     Path("docker-compose.windows.yml").write_text("""version: '3.8' 
# services:
#   web:
#     build: .
#     ports:
#       - "80:80"
#     environment:
#       - DEBUG=true
#       - DB_PASSWORD=secret123
# """) 

    # GitLab CI de test
    Path(".gitlab-ci.yml").write_text("""stages:
  - build
  - deploy

build:
  stage: build
  script:
    - docker build -t myapp .
    - docker push myapp

deploy:
  stage: deploy
  script:
    - kubectl apply -f k8s/
""")
    

    start = time.time()
    results = await analyze_project(test_dir)
    total_time = time.time() - start

    # Afficher les résultats
    for r in results:
        print(f"{'='*50}")
        print(f"📄 {r['file']} [{r['time']:.1f}s]")
        print(f"{'='*50}")
        print(r['analysis'])
        print()

    # Résumé
    print(f"\n{'='*50}")
    print(f"📊 RÉSUMÉ")
    print(f"{'='*50}")
    print(f"Fichiers analysés: {len(results)}")
    print(f"Temps total: {total_time:.2f}s")
    sum_individual = sum(r['time'] for r in results)
    print(f"Temps si séquentiel: ~{sum_individual:.2f}s")
    print(f"Gain: {sum_individual/total_time:.1f}x plus rapide")


asyncio.run(main())