"""
Générer des embeddings pour du texte en utilisant un modèle d'embedding. 
Les embeddings sont des représentations vectorielles de texte qui capturent le 
sens et les relations entre les mots. Ils sont utiles pour des tâches comme la 
recherche sémantique, la classification de texte, la détection de similarité, etc. 
Ce fichier montre comment utiliser l'API pour obtenir des embeddings à partir 
d'une phrase ou d'un document.
Il utilise la fonction embedding pour obtenir un vecteur d'embedding à partir 
d'une entrée textuelle.
Il affiche les dimensions de l'embedding et un échantillon des valeurs du vecteur.
"""


from litellm import embedding
from dotenv import load_dotenv

load_dotenv()

# Génère un embedding pour une question sur la configuration d'un pipeline GitLab CI.
response = embedding(
    model="openai/text-embedding-3-small",
    input=["Comment configurer un pipeline GitLab CI ?"]
)

# Affiche les dimensions et un échantillon des valeurs du vecteur.
vector = response.data[0]["embedding"]
print(f"📐 Dimensions: {len(vector)}")  # 1536
print(f"📊 Échantillon: {vector[:5]}")