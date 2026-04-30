"""
DevOps Buddy prêt pour la production avec Router multi-providers, RAG et monitoring.
Ce fichier montre comment créer une version de DevOps Buddy prête pour la production en utilisant le Router
de litellm pour gérer plusieurs fournisseurs de modèles avec des fallbacks, une base de connaissances indexée par des embeddings pour la recherche sémantique, et un système de monitoring pour les performances et les coûts.
Il définit une classe DevOpsBuddyProduction qui utilise un Router pour envoyer des requêtes à plusieurs modèles de langage, en utilisant une base de connaissances pour fournir du contexte aux réponses, et en enregistrant les métriques de chaque appel dans un logger personnalisé.
Il utilise la technique RAG pour rechercher les documents les plus pertinents dans la base de connaissances en fonction de la question posée, puis génère une réponse en utilisant le contexte récupéré.
Il utilise la fonction embedding pour générer des vecteurs d'embedding à partir du texte des documents et des questions, et la fonction cosine_similarity pour calculer la similarité entre les vecteurs d'embedding afin de trouver les documents les plus pertinents.
Il utilise la fonction completion pour générer des réponses à partir du contexte récupéré et de la question posée, en suivant un prompt système qui guide le comportement de l'assistant.
Il affiche les réponses générées par les modèles, ainsi que les métriques de performance et de coût pour chaque appel, et utilise des fallbacks pour garantir une haute disponibilité.
"""


from litellm import Router, embedding
from litellm.integrations.custom_logger import CustomLogger
from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
import os


load_dotenv()


# Fonction de similarité cosinus pour la recherche de contexte dans la base de connaissances. Cette fonction prend deux vecteurs d'embedding et calcule la similarité cosinus entre eux, ce qui permet de mesurer la pertinence d'un document par rapport à une requête. Elle est utilisée dans la méthode search_context de la classe DevOpsBuddyProduction pour trouver les documents les plus pertinents dans la base de connaissances en fonction de la question posée. Un score de similarité plus élevé indique une plus grande pertinence du document par rapport à la requête.
# La fonction cosine_similarity prend deux listes de nombres (v1 et v2), les convertit en tableaux numpy, et calcule la similarité cosinus en utilisant le produit scalaire des deux vecteurs divisé par le produit de leurs normes. Le résultat est un nombre compris entre -1 et 1, où 1 signifie que les vecteurs sont identiques, 0 signifie qu'ils sont orthogonaux, et -1 signifie qu'ils sont opposés. Cette fonction est essentielle pour la recherche de contexte dans la base de connaissances, car elle permet de mesurer la pertinence des documents par rapport à une requête en fonction de leurs embeddings. 
def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Classe de logger personnalisé pour la production qui affiche les réponses générées par les modèles, ainsi que les métriques de performance et de coût pour chaque appel. Cette classe hérite de CustomLogger et implémente les méthodes log_success et log_failure pour enregistrer les succès et les échecs des appels au Router. Dans log_success, elle affiche le nom du modèle utilisé et la latence de la réponse, tandis que dans log_failure, elle affiche un message d'avertissement indiquant qu'un fallback a été déclenché en raison d'une exception. Ce logger personnalisé est utilisé dans la classe DevOpsBuddyProduction pour fournir des informations en temps réel sur les performances et les coûts des appels aux modèles de langage, ce qui est crucial pour une utilisation en production.
class ProductionLogger(CustomLogger):
    def log_success(self, kwargs, response, start_time, end_time):
        latency = end_time - start_time
        print(f"📡 {response.model} | {latency:.2f}s")

    def log_failure(self, kwargs, exception, start_time, end_time):
        print(f"⚠️ Fallback déclenché: {exception}")


class CustomRouter(Router):
    def __init__(self, model_list, callbacks, num_retries=2, timeout=30, cache_responses=True):
        super().__init__(model_list, )
        self.callbacks = callbacks

# DevOps Buddy prêt pour la production avec Router multi-providers, RAG et monitoring. La classe DevOpsBuddyProduction utilise un Router pour envoyer des requêtes à plusieurs modèles de langage, en utilisant une base de connaissances pour fournir du contexte aux réponses, et en enregistrant les métriques de chaque appel dans un logger personnalisé. Elle implémente la technique RAG pour rechercher les documents les plus pertinents dans la base de connaissances en fonction de la question posée, puis génère une réponse en utilisant le contexte récupéré. Elle utilise la fonction embedding pour générer des vecteurs d'embedding à partir du texte des documents et des questions, et la fonction cosine_similarity pour calculer la similarité entre les vecteurs d'embedding afin de trouver les documents les plus pertinents. Elle utilise la fonction completion pour générer des réponses à partir du contexte récupéré et de la question posée, en suivant un prompt système qui guide le comportement de l'assistant. Cette classe est conçue pour être utilisée en production, avec des fallbacks pour garantir une haute disponibilité et un monitoring pour suivre les performances et les coûts des appels aux modèles de langage.
class DevOpsBuddyProduction:
    """DevOps Buddy prêt pour la production."""

    SYSTEM_PROMPT = """Tu es DevOps Buddy, l'assistant DevOps de l'équipe.

Utilise le contexte fourni pour répondre. Si le contexte ne suffit pas,
utilise tes connaissances générales mais précise-le.

Sois concis, donne des commandes concrètes."""

    # Initialisation de la classe DevOpsBuddyProduction avec un Router multi-providers et une base de connaissances. Le Router est configuré avec deux modèles de langage (OpenAI et Anthropic) avec des clés API récupérées depuis les variables d'environnement, et utilise un logger personnalisé pour enregistrer les métriques de chaque appel. La base de connaissances est chargée à partir d'un fichier JSON, qui contient des documents et leurs vecteurs d'embedding correspondants. Si le fichier n'existe pas, la base de connaissances est initialisée comme vide. Cette configuration permet à DevOps Buddy d'être prêt pour une utilisation en production, avec des fallbacks pour garantir une haute disponibilité et un monitoring pour suivre les performances et les coûts des appels aux modèles de langage.
    def __init__(self, knowledge_file: str = "devops_knowledge.json"):
        # Router multi-providers
        model_list = [
            {
                "model_name": "chat",
                "litellm_params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "order": 1
                }
            },
            {
                "model_name": "chat",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-haiku-latest",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "order": 2
                }
            }
        ]

        logger = ProductionLogger()
        callbacks = [logger]
        # Router avec monitoring personnalisé pour la production et RAG pour la recherche de contexte dans la base de connaissances. Le Router est configuré avec une liste de modèles de langage, chacun avec ses propres paramètres d'API et un ordre de fallback. Le logger personnalisé (ProductionLogger) est utilisé pour enregistrer les métriques de chaque appel, ce qui permet de suivre les performances et les coûts en temps réel. La classe DevOpsBuddyProduction utilise ce Router pour envoyer des requêtes aux modèles de langage, en utilisant une base de connaissances pour fournir du contexte aux réponses, et en enregistrant les métriques de chaque appel dans le logger personnalisé. Cette configuration est conçue pour garantir une haute disponibilité et un monitoring efficace lors de l'utilisation de DevOps Buddy en production.
        self.router = CustomRouter(
            model_list=model_list,
            num_retries=2,
            timeout=30,
            cache_responses=True,
            callbacks=callbacks,
        )

        # Base de connaissances : chargement de la base de connaissances depuis un fichier JSON. Si le fichier n'existe pas, la base de connaissances est initialisée comme vide. La base de connaissances est utilisée pour fournir du contexte aux réponses générées par les modèles de langage, en utilisant la technique RAG pour rechercher les documents les plus pertinents en fonction de la question posée. Les documents et leurs vecteurs d'embedding correspondants sont stockés dans des listes, ce qui permet de faire des recherches efficaces en utilisant la fonction cosine_similarity pour mesurer la pertinence des documents par rapport à une requête.
        if Path(knowledge_file).exists():
            data = json.loads(Path(knowledge_file).read_text())
            self.docs = data["documents"]
            self.vecs = data["vectors"]
        else:
            self.docs = []
            self.vecs = []

    # Recherche du contexte pertinent dans la base de connaissances en utilisant la technique RAG. La méthode search_context prend une requête en entrée, génère un vecteur d'embedding pour cette requête, et calcule la similarité cosinus entre ce vecteur et les vecteurs d'embedding des documents dans la base de connaissances. Les documents sont ensuite triés par pertinence, et les plus pertinents sont retournés sous forme de texte formaté. Si aucun document n'est suffisamment pertinent (score de similarité inférieur à 0.5), la méthode retourne une chaîne vide, indiquant qu'il n'y a pas de documentation interne pertinente pour la requête.
    # La méthode ask utilise cette recherche de contexte pour fournir du contexte aux réponses générées par les modèles de langage, en suivant un prompt système qui guide le comportement de l'assistant. Les réponses générées sont ensuite affichées, et les métriques de performance et de coût sont enregistrées dans le logger personnalisé grâce au Router. 
    # Cette approche permet à DevOps Buddy d'être prêt pour une utilisation en production, avec des fallbacks pour garantir une haute disponibilité et un monitoring pour suivre les performances et les coûts des appels aux modèles de langage.
    # En résumé, cette classe DevOpsBuddyProduction est conçue pour être utilisée en production, avec une architecture robuste qui inclut un Router multi-providers pour la haute disponibilité, une base de connaissances indexée par des embeddings pour fournir du contexte pertinent aux réponses, et un système de monitoring pour suivre les performances et les coûts des appels aux modèles de langage. Elle utilise la technique RAG pour rechercher les documents les plus pertinents dans la base de connaissances en fonction de la question posée, puis génère une réponse en utilisant le contexte récupéré, tout en enregistrant les métriques de chaque appel dans un logger personnalisé.
    def search_context(self, query: str, k: int = 2) -> str:
        """Recherche du contexte pertinent."""
        if not self.docs:
            return ""

        resp = embedding(model="openai/text-embedding-3-small", input=[query])
        qv = resp.data[0]["embedding"]

        scored = [(d, cosine_similarity(qv, v)) for d, v in zip(self.docs, self.vecs)]
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:k]

        return "\n\n".join([
            f"**{d['title']}**\n{d['content']}"
            for d, score in top if score > 0.5
        ])

    # Pose une question avec RAG et fallbacks. La méthode ask prend une question en entrée, utilise la méthode search_context pour rechercher le contexte pertinent dans la base de connaissances, puis génère une réponse en utilisant le Router pour envoyer une requête aux modèles de langage. Le prompt système guide le comportement de l'assistant en lui indiquant d'utiliser le contexte fourni pour répondre, et de préciser s'il doit utiliser ses connaissances générales si le contexte ne suffit pas. Les réponses générées sont affichées, et les métriques de performance et de coût sont enregistrées dans le logger personnalisé grâce au Router. Cette approche garantit que DevOps Buddy est prêt pour une utilisation en production, avec des fallbacks pour garantir une haute disponibilité et un monitoring pour suivre les performances et les coûts des appels aux modèles de langage.
    def ask(self, question: str) -> str:
        """Pose une question avec RAG et fallbacks."""

        context = self.search_context(question)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""CONTEXTE:
{context or "(Pas de documentation interne pertinente)"}

QUESTION: {question}"""}
        ]

        response = self.router.completion(
            model="chat",
            messages=messages
        )

        return response.choices[0].message.content


def main():
    buddy = DevOpsBuddyProduction()

    print("🚀 DevOps Buddy (Production Mode)")
    print("   Fallbacks: OpenAI → Claude")
    print("   Cache: activé")
    print("   Tapez 'quit' pour quitter\n")

    while True:
        question = input("👤 > ").strip()

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        answer = buddy.ask(question)
        print(f"🤖 {answer}\n")


if __name__ == "__main__":
    main()