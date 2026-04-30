FROM ubuntu:latest # Image de base Ubuntu latest avec les derniers paquets et mises à jour de sécurité dans le conteneur de base pour assurer une meilleure sécurité et compatibilité. 
RUN apt-get update && apt-get install -y python3 curl wget git
COPY . /app
RUN chmod 777 /app
CMD ["python3", "app.py"]
