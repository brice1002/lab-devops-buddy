# Ce fichier analyse un fichier de configuration DevOps (comme un Dockerfile) en utilisant le modèle GPT-4.1-mini pour identifier les problèmes de sécurité, suggérer des améliorations de performance et vérifier les bonnes pratiques. Le résultat est formaté de manière structurée pour faciliter la lecture et l'interprétation.


import asyncio
from pathlib import Path
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """Tu es DevOps Buddy, un auditeur expert en fichiers de configuration DevOps.

Pour chaque fichier, tu dois :
1. Identifier le type de fichier (Dockerfile, CI, Terraform, etc.)
2. Lister les problèmes de sécurité
3. Suggérer des améliorations de performance
4. Vérifier les bonnes pratiques

Format de réponse :
## Type: [type]
## Problèmes: [liste ou "Aucun"]
## Améliorations: [liste ou "Aucune"]
## Score: [1-10]"""


async def analyze_file(filepath: Path) -> dict:
    """Analyse un fichier DevOps et retourne un rapport."""

    content = filepath.read_text()

    response = await acompletion(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyse ce fichier '{filepath.name}':\n\n```\n{content}\n```"}
        ],
        max_tokens=500
    )

    return {
        "file": filepath.name,
        "analysis": response.choices[0].message.content,
        "tokens": response.usage.total_tokens
    }


# async def main():
#     # Exemple avec un Dockerfile
#     dockerfile = Path("Dockerfile")
#     if not dockerfile.exists():
#         dockerfile.write_text("""FROM python:3.11
# RUN pip install flask
# COPY . /app
# WORKDIR /app
# CMD ["python", "app.py"]
# """)

#     result = await analyze_file(dockerfile)
#     print(f"📄 {result['file']}")
#     print(result['analysis'])
#     print(f"\n📊 Tokens: {result['tokens']}")


# asyncio.run(main())

async def main():
    # Exemple avec un Dockerfile
    fichier = Path("06_async_basic.py")
    # dockerfile = Path("Dockerfile")
    if not fichier.exists():
        fichier.write_text("""FROM python:3.11
RUN pip install flask
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
""")

    result = await analyze_file(fichier)
    print(f"📄 {result['file']}")
    print(result['analysis'])
    print(f"\n📊 Tokens: {result['tokens']}")


asyncio.run(main())

