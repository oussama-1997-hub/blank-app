# app_personalized_recommendation.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lean 4.0 Diagnostic", layout="wide")
st.title("🏭 Évaluation Personnalisée de la Maturité Lean 4.0")

st.markdown("""
Cette application vous permet de :
1. Évaluer votre positionnement Lean 4.0
2. Comparer vos outils avec les leaders
3. Générer une feuille de route personnalisée
""")

# --- Upload CSV file ---
file = st.sidebar.file_uploader("📂 Importer vos données Lean 4.0 (CSV)", type="csv")

if file:
    df = pd.read_csv(file)

    colonnes = [col for col in df.columns if "Leadership" in col or "Supply" in col or "Opérations" in col or "Technologies" in col or "Organisation" in col]
    dummy_cols = [col for col in df.columns if col.startswith("Tech_") or col.startswith("Lean_")]

    st.success("✅ Données chargées avec succès")

    # Step 1: Clustering
    features_cluster = df[colonnes].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_cluster)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled)

    labels_map = {0: "Initial", 1: "Émergent", 2: "Avancé"}
    df['Niveau Observé'] = df['cluster'].map(labels_map)

    # Step 2: Decision Tree
    X = df[dummy_cols].fillna(0)
    y = df['Niveau Observé']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = DecisionTreeClassifier(min_samples_leaf=3, random_state=42)
    clf.fit(X_train, y_train)
    df['Niveau Prévu'] = clf.predict(X)

    # Step 3: Recommandation logiques
    st.header("📊 Résultats du Diagnostic")
    for i, row in df.iterrows():
        st.markdown(f"**Entreprise {i+1}**")
        obs = row['Niveau Observé']
        pre = row['Niveau Prévu']
        st.write(f"- Niveau Observé (KMeans): `{obs}`")
        st.write(f"- Niveau Prévu (Arbre): `{pre}`")

        if obs != pre:
            if obs == "Avancé" and pre != "Avancé":
                st.warning("🔺 Vous avez un bon niveau mais peu d'outils. Priorisez les technologies utilisées par les entreprises avancées.")
            elif pre == "Avancé" and obs != "Avancé":
                st.error("🔻 Vous avez des outils avancés mais une organisation peu mature. Concentrez-vous sur la stratégie et la formation.")
            else:
                st.info("🔄 Écart intermédiaire. Améliorez à la fois outils et sous-dimensions.")
        else:
            st.success("✅ Vous êtes cohérent avec votre niveau de maturité.")

    # Step 4: Visuals
    st.header("📈 Visualisation des Scores Moyens par Cluster")
    avg = df.groupby('cluster')[colonnes].mean()
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(avg.T, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.header("🚀 Feuille de Route Personnalisée")
    st.markdown("Suggestions d’outils et pratiques à adopter basées sur les écarts constatés :")
    for cl in sorted(df['cluster'].unique()):
        st.subheader(f"Cluster {cl} - {labels_map[cl]}")
        tech_avg = df[df['cluster'] == cl][dummy_cols].mean().sort_values(ascending=False).head(10)
        st.dataframe(tech_avg)

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
