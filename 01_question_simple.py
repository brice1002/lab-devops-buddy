from dotenv import load_dotenv
from litellm import completion, completion_cost

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Notre première question DevOps
response = completion(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explique la différence entre Docker et une VM en 3 lignes."}
    ]
)

# Afficher la réponse de l'IA
print(response.choices[0].message.content)

# Afficher le modèle utilisé
print(f"Modèle utilisé : {response.model}")

# Les tokens utilisés pour la question
print(f"Tokens utilisés pour la question : {response.usage.prompt_tokens} tokens")

# Les tokens utilisés pour la réponse
print(f"Tokens utilisés pour la réponse : {response.usage.completion_tokens} tokens")

# Afficher le prix de la demande en tokens
print(f"Prix de la demande : {response.usage.total_tokens} tokens")

# # Afficher le coût estimé de la demande
print(f"Tokens: {response.usage.total_tokens}")
print(f"Coût estimé: ${response._hidden_params.get('response_cost', 0):.4f}")