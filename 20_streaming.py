from litellm import completion
from dotenv import load_dotenv

load_dotenv()

response = completion(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "Liste 5 bonnes pratiques pour un Dockerfile"}
    ],
    # stream permet de recevoir la réponse en temps réel, au fur et à mesure que l'IA la génère
    stream=True  # Active le streaming
)

# Afficher la réponse en temps réel. Cette boucle s'exécutera à chaque fois que l'IA génère une nouvelle partie de la réponse
for chunk in response:
    # chunk.choices[0].delta contient la nouvelle partie de la réponse générée par l'IA
    content = chunk.choices[0].delta.content or ""
    # Afficher la nouvelle partie de la réponse sans sauter de ligne, et forcer l'affichage immédiat
    print(content, end="", flush=True)

print()  # Nouvelle ligne finale