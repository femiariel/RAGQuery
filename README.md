# RAGQuery

Ce projet développe un chatbot capable d’interagir et de répondre à des questions spécifiques basées sur le contenu d’un fichier PDF. Le processus implique plusieurs étapes et technologies clés :

## 1. Extraction de Texte avec OCR et PyMuPDF
- **Utilisation de PyMuPDF (Fitz)** : pour lire et extraire le texte directement des pages PDF lorsque possible.
- **Application de la technologie OCR (EasyOCR)** : pour extraire du texte des images contenues dans les PDFs, permettant de récupérer des informations même des documents numérisés ou avec contenu graphique.

## 2. Traitement de Texte avec NLTK
- **Nettoyage du texte extrait** : pour éliminer les sauts de ligne et espaces superflus.
- **Utilisation de NLTK** : pour la segmentation du texte en phrases, facilitant l’analyse et le traitement des données textuelles à un niveau plus granulaire.

## 3. Génération de Vecteurs avec Sentence Transformers
- **Conversion des phrases extraites en vecteurs numériques** : via un modèle de Sentence Transformers, permettant une représentation dense du contenu textuel pour des comparaisons ultérieures.

## 4. Indexation et Recherche avec FAISS
- **Création d’un index avec FAISS** : pour les vecteurs de phrases, optimisant la recherche de contenu pertinent basé sur des requêtes d’entrée.
- **Fonctionnalité de recherche dans l’index** : pour identifier les phrases les plus pertinentes en fonction de la question posée.

## 5. Intégration avec LangChain et dolly-v2-3b LLM
- **Utilisation du modèle de langage grand modèle dolly-v2-3b** : à travers LangChain pour générer des réponses informatives.
- **LangChain** : facilite l’intégration des modèles de langage avec des processus de récupération d’informations, améliorant la qualité des réponses générées.

## 6. Interface Utilisateur avec Gradio
- **Déploiement d’une interface utilisateur interactive via Gradio** : permettant aux utilisateurs de soumettre des fichiers PDF et des questions, et de recevoir des réponses générées par le système.
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
