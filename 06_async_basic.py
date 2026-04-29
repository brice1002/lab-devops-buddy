# Ce fichier montre comment utiliser l'API de manière asynchrone avec asyncio, permettant d'envoyer des requêtes sans bloquer le programme. Il utilise la fonction acompletion pour obtenir une réponse du modèle de manière asynchrone.


import asyncio
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

async def main():
    response = await acompletion(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Qu'est-ce qu'un multi-stage build Docker ?"}
        ]
    )
    print(response.choices[0].message.content)

asyncio.run(main())