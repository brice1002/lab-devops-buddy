# ce fichier contient une conversation interactive avec le modèle, en utilisant un contexte de conversation pour maintenir l'état entre les messages.

from litellm import completion
from dotenv import load_dotenv

load_dotenv()


# Personnalité de DevOps Buddy
# Le system prompt définit le contexte et les instructions pour le modèle, lui indiquant qu'il est un assistant expert en DevOps et DevSecOps, avec des spécialités spécifiques. Cela aide à orienter les réponses du modèle pour qu'elles soient pertinentes et adaptées au domaine DevOps.
SYSTEM_PROMPT = """Tu es DevOps Buddy, un assistant expert en DevOps et DevSecOps.

Tes spécialités :
- CI/CD (GitLab, GitHub Actions, Jenkins)
- Conteneurs (Docker, Kubernetes, Podman)
- Infrastructure as Code (Terraform, Ansible)
- Observabilité (Prometheus, Grafana, Loki)
- Sécurité (scanning, SAST/DAST, secrets)

Réponds de manière concise et pratique. Donne des exemples de code quand pertinent."""

# Fonction pour envoyer des messages et obtenir des réponses du modèle, en maintenant le contexte de la conversation.
def chat(messages: list, user_input: str) -> str:
    """Envoie un message et retourne la réponse."""
    messages.append({"role": "user", "content": user_input})

    response = completion(
        model="gpt-4.1-mini",
        messages=messages,
        stream=True
    )

    # Afficher la réponse en temps reel et construire la réponse complète au fur et à mesure
    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n")
    messages.append({"role": "assistant", "content": full_response})
    return full_response

# Fonction principale pour gérer la boucle de conversation avec l'utilisateur
def main():
    print("🤖 DevOps Buddy v1.0")
    print("Posez vos questions DevOps. Tapez 'quit' pour quitter.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("👤 Vous: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 À bientôt !")
            break
        if not user_input:
            continue

        print("\n🤖 DevOps Buddy: ", end="")
        chat(messages, user_input)

if __name__ == "__main__":
    main()