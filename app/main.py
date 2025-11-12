# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from typing import List
import numpy as np
from . import model
from . import plot_clusters

app = FastAPI(title="API Clustering avec CSV")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/plot_csv", tags=["Clustering"])
async def get_cluster_plot_from_csv(file: UploadFile = File(...)):
    # Lire le fichier CSV
    df = pd.read_csv(file.file, sep=";")
    
    # Vérifier que le fichier contient uniquement des colonnes numériques
    if not all(df.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))):
        return {"error": "Le fichier doit contenir uniquement des colonnes numériques."}
    
    # Convertir en tableau numpy
    X = df.to_numpy()
    
    # Appliquer le modèle
    labels = model.predict(X)
    
    # Retourner le graphique
    return plot_clusters(X, labels)