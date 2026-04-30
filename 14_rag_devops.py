"""
RAG (Retrieval-Augmented Generation) pour un assistant DevOps avec une base de connaissances interne.
Ce fichier montre comment créer un assistant DevOps qui utilise une base de connaissances interne pour 
répondre à des questions techniques. L'assistant utilise la technique RAG pour rechercher des documents 
les plus pertinents dans la base de connaissances en fonction de la question posée, puis génère une 
réponse en utilisant le contexte récupéré.
Il définit une classe DevOpsBuddy qui encapsule la logique de recherche et de génération de réponses. 
La classe charge une base de connaissances à partir d'un fichier JSON, effectue des recherches sémantiques 
pour trouver les documents les plus pertinents, et utilise un modèle de langage pour générer des réponses 
basées sur le contexte récupéré.
Il utilise la fonction embedding pour générer des vecteurs d'embedding à partir du texte des documents et 
des questions, et la fonction cosine_similarity pour calculer la similarité entre les vecteurs d'embedding 
afin de trouver les documents les plus pertinents.
Il utilise la fonction completion pour générer des réponses à partir du contexte récupéré et de la question 
posée, en suivant un prompt système qui guide le comportement de l'assistant.
"""


from litellm import embedding, completion
from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class DevOpsBuddy:
    """Assistant DevOps avec base de connaissances RAG."""

    SYSTEM_PROMPT = """Tu es DevOps Buddy, un assistant DevOps expert.

Tu as accès à une base de connaissances interne. Utilise UNIQUEMENT
les informations fournies dans le contexte pour répondre. Si le contexte
ne contient pas l'information nécessaire, dis-le clairement.

Réponds de manière concise et actionnable avec des commandes concrètes."""

    def __init__(self, knowledge_file: str = "devops_knowledge.json"):
        self.model = "openai/text-embedding-3-small"
        self.llm = "gpt-4.1-mini"

        # Charger la base de connaissances
        data = json.loads(Path(knowledge_file).read_text())
        self.documents = data["documents"]
        self.vectors = data["vectors"]

    def search(self, query: str, top_k: int = 2) -> list[dict]:
        """Recherche dans la base de connaissances."""
        response = embedding(model=self.model, input=[query])
        query_vector = response.data[0]["embedding"]

        scored = []
        for doc, vec in zip(self.documents, self.vectors):
            score = cosine_similarity(query_vector, vec)
            if score > 0.5:  # Seuil de pertinence
                scored.append({"score": score, **doc})

        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def ask(self, question: str) -> str:
        """Pose une question avec contexte RAG."""

        # Rechercher le contexte pertinent
        context_docs = self.search(question)

        # Construire le contexte
        if context_docs:
            context = "\n\n---\n\n".join([
                f"**{d['title']}**\n{d['content']}"
                for d in context_docs
            ])
            sources = [d['title'] for d in context_docs]
        else:
            context = "(Aucun document pertinent trouvé)"
            sources = []

        # Appeler le LLM avec le contexte
        response = completion(
            model=self.llm,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""CONTEXTE (documentation interne):
{context}

QUESTION: {question}"""}
            ]
        )

        answer = response.choices[0].message.content

        # Ajouter les sources
        if sources:
            answer += f"\n\n📚 Sources: {', '.join(sources)}"

        return answer


# Test
buddy = DevOpsBuddy()

questions = [
    "Mon pod Kubernetes est en CrashLoopBackOff, que faire ?",
    "Comment déployer avec ArgoCD ?",
    "Le pipeline GitLab est bloqué depuis 30 minutes"
]

for q in questions:
    print(f"👤 {q}")
    print(f"🤖 {buddy.ask(q)}")
    print("\n" + "="*60 + "\n")