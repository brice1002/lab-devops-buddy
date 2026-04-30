"""
RAG (Retrieval-Augmented Generation) pour DevOps Buddy. 
Ce fichier montre comment créer un assistant DevOps qui utilise une base 
de connaissances interne pour répondre à des questions techniques. 
L'assistant utilise la technique RAG pour rechercher des documents les 
plus pertinents dans la base de connaissances en fonction de la question posée, 
puis génère une réponse en utilisant le contexte récupéré.
Il définit une classe DevOpsBuddy qui encapsule la logique de recherche et de génération de réponses.
La classe charge une base de connaissances à partir d'un fichier JSON, effectue des recherches sémantiques pour trouver les documents les plus pertinents, et utilise un modèle de langage pour générer des réponses basées sur le contexte récupéré.
Il utilise la fonction embedding pour générer des vecteurs d'embedding à partir du texte des documents et des questions, et la fonction cosine_similarity pour calculer la similarité entre les vecteurs d'embedding afin de trouver les documents les plus pertinents.
Il utilise la fonction completion pour générer des réponses à partir du contexte récupéré et de la question posée, en suivant un prompt système qui guide le comportement de l'assistant.
"""


from litellm import embedding, completion
from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class DevOpsBuddy:
    """Assistant DevOps avec RAG."""

    SYSTEM = """Tu es DevOps Buddy. Réponds en utilisant le contexte fourni.
Si le contexte ne suffit pas, dis-le. Sois concis et donne des commandes."""

    def __init__(self, kb_file: str = "devops_knowledge.json"):
        data = json.loads(Path(kb_file).read_text())
        self.docs = data["documents"]
        self.vecs = data["vectors"]

    def search(self, query: str, k: int = 2):
        resp = embedding(model="openai/text-embedding-3-small", input=[query])
        qv = resp.data[0]["embedding"]
        scored = [(d, cosine_similarity(qv, v)) for d, v in zip(self.docs, self.vecs)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    def ask(self, question: str) -> str:
        results = self.search(question)
        context = "\n\n".join([f"**{d['title']}**\n{d['content']}" for d, _ in results])

        resp = completion(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"CONTEXTE:\n{context}\n\nQUESTION: {question}"}
            ]
        )
        return resp.choices[0].message.content


def main():
    buddy = DevOpsBuddy()

    print("🤖 DevOps Buddy (tapez 'quit' pour quitter)\n")

    while True:
        question = input("👤 > ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("👋 À bientôt !")
            break

        if not question:
            continue

        print(f"🤖 {buddy.ask(question)}\n")


if __name__ == "__main__":
    main()