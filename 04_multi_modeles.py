# Ce fichier compare les réponses de différents modèles à la même question, en affichant les réponses et le nombre de tokens utilisés pour chaque modèle.


from litellm import completion
from dotenv import load_dotenv

load_dotenv()

QUESTION = "Donne 3 bonnes pratiques pour écrire un Dockerfile"

MODELES = [
    "gpt-4.1-mini",
    "claude-3-5-haiku-latest",
    # "ollama/llama3.2",  # Décommentez si Ollama est installé
]

for modele in MODELES:
    print(f"\n{'='*50}")
    print(f"🤖 Réponse de {modele}")
    print('='*50)

    try:
        response = completion(
            model=modele,
            messages=[{"role": "user", "content": QUESTION}],
            max_tokens=300
        )
        print(response.choices[0].message.content)
        print(f"\n📊 Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Erreur: {e}")