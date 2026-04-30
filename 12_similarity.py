"""
Ce fichier montre comment utiliser l'API pour obtenir des embeddings à partir 
d'une phrase ou d'un document et calculer la similarité cosinus entre deux 
embeddings.
Il utilise la fonction embedding pour obtenir un vecteur d'embedding à partir 
d'une entrée textuelle.
Il affiche la question et les documents DevOps à comparer, ainsi que la similarité cosinus entre l'embedding de la question et celui de chaque document.
Il utilise la bibliothèque numpy pour calculer la similarité cosinus entre les vecteurs d'embedding.
"""


from litellm import embedding
from dotenv import load_dotenv
import numpy as np

load_dotenv()


# Fonction pour calculer la similarité cosinus entre deux vecteurs d'embedding numpy arrays. La similarité cosinus est une mesure de la similarité entre deux vecteurs dans un espace vectoriel, calculée comme le cosinus de l'angle entre eux. Elle varie entre -1 (opposés) et 1 (identiques), avec 0 indiquant une orthogonalité (pas de similarité).
def cosine_similarity(v1: list, v2: list) -> float:
    """Similarité cosinus entre deux vecteurs."""
    a = np.array(v1)
    b = np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Documents DevOps à comparer
docs = [
    "ArgoCD déploie automatiquement depuis Git",
    "GitOps synchronise le cluster avec le repo",
    "Docker build crée une image de conteneur"
]

response = embedding(
    model="openai/text-embedding-3-small",
    input=docs
)

vectors = [item["embedding"] for item in response.data]

# Chercher le doc le plus proche de la question
question = "comment automatiser les déploiements k8s"
q_response = embedding(
    model="openai/text-embedding-3-small",
    input=[question]
)
q_vector = q_response.data[0]["embedding"]

print(f"🔍 Question: {question}\n")
for doc, vec in zip(docs, vectors):
    score = cosine_similarity(q_vector, vec)
    print(f"  [{score:.3f}] {doc}")