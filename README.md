# chatpdf
Ce projet a pour but de mettre en place un chatbot qui réponds à toutes vos questions en se basant sur un pdf donnée en entrée par l'utilisateur. Le chatbot utilise le modèle dolly-v2-3b de Databricks et la bibliothèque FAISS pour le Retrieval Augmented Generation.  Pour faire plus simple , c'est une implémentation de RAG couplé avec un LLM.

## Projet en local
Une fois le projet cloné en local, utiliser la commande
```
pip install  -r requirements.txt 
```
pour installer toutes les bibliothèques nécessaires au bon fonctionnement du projet.
Ensuite vous n'aurez qu'a exécuter le fichier ari-3b.py en utilisant la commande 
```
python ari-3b.py
```
et vous aurez accès à une interface graphique simple réaliser avec gradio. Une fois sur l'interface , mettez en entrée votre fichier au format pdf et poser les questions que vous souhaitez sur le fichier et vous aurez des réponses.
