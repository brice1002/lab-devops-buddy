"""
Indexer une base de connaissances DevOps avec des embeddings pour la recherche sémantique.
Ce fichier montre comment créer une base de connaissances simple en utilisant des embeddings 
pour indexer des documents et effectuer des recherches sémantiques.
Il définit une classe KnowledgeBase qui permet d'ajouter des documents avec un titre, un contenu 
et des tags, et de rechercher les documents les plus pertinents pour une requête donnée.
Il utilise la fonction embedding pour générer des vecteurs d'embedding à partir du texte des 
documents et des requêtes, et la fonction cosine_similarity pour calculer la similarité entre les vecteurs d'embedding.
Il permet également de sauvegarder et de charger la base de connaissances depuis un fichier JSON.
"""


from litellm import embedding
from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# Ce fichier montre comment créer une base de connaissances simple en utilisant des embeddings pour indexer des documents et effectuer des recherches sémantiques.
# Il définit une classe KnowledgeBase qui permet d'ajouter des documents avec un titre, un contenu et des tags, et de rechercher les documents les plus pertinents pour une requête donnée.
# Il utilise la fonction embedding pour générer des vecteurs d'embedding à partir du texte des documents et des requêtes, et la fonction cosine_similarity pour calculer la similarité entre les vecteurs d'embedding.
# Il permet également de sauvegarder et de charger la base de connaissances depuis un fichier JSON.
def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# La classe KnowledgeBase permet de créer une base de connaissances DevOps avec des documents indexés par des embeddings, et de rechercher les documents les plus pertinents pour une requête donnée.
class KnowledgeBase:
    """Base de connaissances DevOps avec recherche sémantique."""

# Créer une base de connaissances avec un modèle d'embedding spécifique. La classe KnowledgeBase utilise un modèle d'embedding pour générer des vecteurs à partir du texte des documents et des requêtes. Par défaut, elle utilise le modèle "openai/text-embedding-3-small", mais cela peut être personnalisé lors de l'instanciation de la classe en faisant par exemple kb = KnowledgeBase(model="gpt-4.1-mini").
    def __init__(self, model: str = "openai/text-embedding-3-small"):
        self.model = model
        self.documents: list[dict] = []
        self.vectors: list[list[float]] = []

    def add_document(self, title: str, content: str, tags: list[str] = None):
        """Ajoute un document à la base."""
        # Créer le texte à indexer
        text = f"{title}\n\n{content}"

        # Générer l'embedding
        response = embedding(model=self.model, input=[text])
        vector = response.data[0]["embedding"]

        self.documents.append({
            "title": title,
            "content": content,
            "tags": tags or []
        })
        self.vectors.append(vector)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Recherche les documents les plus pertinents."""
        # Embedding de la requête
        response = embedding(model=self.model, input=[query])
        query_vector = response.data[0]["embedding"]

        # Calculer les scores
        scored = []
        for doc, vec in zip(self.documents, self.vectors):
            score = cosine_similarity(query_vector, vec)
            scored.append({"score": score, **doc})

        # Trier par pertinence
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def save(self, path: str):
        """Sauvegarde l'index sur disque."""
        data = {
            "documents": self.documents,
            "vectors": self.vectors
        }
        Path(path).write_text(json.dumps(data))

    def load(self, path: str):
        """Charge l'index depuis le disque."""
        data = json.loads(Path(path).read_text())
        self.documents = data["documents"]
        self.vectors = data["vectors"]


# Créer et remplir la base
kb = KnowledgeBase()

kb.add_document(
    title="Runbook: Déploiement ArgoCD",
    content="""Pour déployer une application avec ArgoCD:
1. Pusher les manifests dans le repo Git (branche main)
2. ArgoCD détecte automatiquement le changement
3. Vérifier le statut: argocd app get <app-name>
4. Forcer la sync si nécessaire: argocd app sync <app-name>
5. Rollback si problème: argocd app rollback <app-name>""",
    tags=["argocd", "deploy", "gitops"]
)

kb.add_document(
    title="Runbook: Debug pods Kubernetes",
    content="""Diagnostic d'un pod en erreur:
1. Vérifier le status: kubectl get pods -n <namespace>
2. Voir les événements: kubectl describe pod <pod-name>
3. Logs du conteneur: kubectl logs <pod-name> -f
4. Shell dans le pod: kubectl exec -it <pod-name> -- /bin/sh
5. Métriques: kubectl top pod <pod-name>""",
    tags=["kubernetes", "debug", "pods"]
)

kb.add_document(
    title="Runbook: Pipeline GitLab CI bloqué",
    content="""Si un pipeline GitLab CI est bloqué:
1. Vérifier les runners disponibles: Settings > CI/CD > Runners
2. Consulter les logs du job: cliquer sur le job en échec
3. Variables manquantes: Settings > CI/CD > Variables
4. Relancer le job: bouton "Retry"
5. Cache corrompue: vider via Settings > CI/CD > Clear Runner Caches""",
    tags=["gitlab", "ci", "pipeline"]
)

kb.add_document(
    title="Bonnes pratiques Dockerfile",
    content="""Optimisations Dockerfile:
1. Utiliser des images légères (alpine, distroless)
2. Multi-stage builds pour réduire la taille finale
3. Un seul RUN pour apt-get update && install && clean
4. USER non-root pour la sécurité
5. HEALTHCHECK pour la supervision
6. Copier les dépendances AVANT le code (cache)""",
    tags=["docker", "dockerfile", "best-practices"]
)

# Sauvegarder
kb.save("devops_knowledge.json")
print(f"✅ {len(kb.documents)} documents indexés")

# Tester la recherche
results = kb.search("mon pod kubernetes ne démarre pas")
print("\n🔍 Recherche: 'mon pod kubernetes ne démarre pas'\n")
for r in results:
    print(f"  [{r['score']:.3f}] {r['title']}")