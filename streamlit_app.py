# streamlit_cluster_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

st.set_page_config(page_title="Lean 4.0 Cluster & Tree App", layout="wide")
st.title("🔍 Lean 4.0 Clustering & Decision Tree Dashboard")
st.markdown("This app lets you perform clustering and classification on Lean 4.0 survey data using KMeans and Decision Tree models.")

# --- Sidebar Config ---
st.sidebar.header("📂 Upload your CSV file")
file = st.sidebar.file_uploader("Upload df_cleaned_with_dummies.csv", type="csv")

# --- Define the column names ---
colonnes = [
    "Leadership - Engagement Lean ",
    "Leadership - Engagement DT",
    "Leadership - Stratégie ",
    "Leadership - Communication",
    "Supply Chain - Collaboration inter-organisationnelle",
    "Supply Chain - Traçabilité",
    "Supply Chain - Impact sur les employées",
    "Opérations - Standardisation des processus",
    "Opérations - Juste-à-temps (JAT)",
    "Opérations - Gestion des résistances",
    "Technologies - Connectivité et gestion des données",
    "Technologies - Automatisation",
    "Technologies - Pilotage du changement",
    "Organisation apprenante  - Formation et développement des compétences",
    "Organisation apprenante  - Collaboration et Partage des Connaissances",
    "Organisation apprenante  - Flexibilité organisationnelle"
]

cols_to_exclude = [
    'Indicateurs suivis', 'Zone investissement principale',
    'Typologie de production', 'Type de flux', 'Pays ',
    'Méthodes Lean ', 'Technologies industrie 4.0', 'cluster',
    'Cluster', 'Feature_Cluster', 'Niveau Maturité', 'Cluster Label'
] + colonnes

if file:
    df = pd.read_csv(file)
    st.success("✅ File loaded successfully")

    # Drop NaNs and scale features
    features = df[colonnes].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Clustering Section ---
    st.header("📊 Clustering (KMeans)")
    k_range = st.slider("Select range for K", 2, 10, (2, 6))

    inertia, silhouette_scores = [], []
    for k in range(k_range[0], k_range[1] + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled_features)
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, model.labels_))

    st.subheader("Elbow Method")
    fig1, ax1 = plt.subplots()
    ax1.plot(range(k_range[0], k_range[1] + 1), inertia, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    st.pyplot(fig1)

    st.subheader("Silhouette Score")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(k_range[0], k_range[1] + 1), silhouette_scores, marker='o')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Score')
    st.pyplot(fig2)

    final_k = st.selectbox("Select final K", list(range(k_range[0], k_range[1] + 1)))
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    st.subheader("📋 Cluster Analysis")
    st.write(df['cluster'].value_counts())

    # Show Niveau Maturité associated with each cluster
    if 'Niveau Maturité' in df.columns:
        maturite_table = df.groupby('cluster')['Niveau Maturité'].value_counts().unstack(fill_value=0)
        st.subheader("🔎 Distribution of 'Niveau Maturité' per Cluster")
        st.dataframe(maturite_table)

    # Show average survey scores per cluster with YlGnBu color gradient
    avg_scores_per_cluster = df.groupby('cluster')[colonnes].mean()
    st.subheader("📈 Average Survey Scores per Cluster")
    st.dataframe(avg_scores_per_cluster.style.background_gradient(cmap="YlGnBu", axis=1, vmin=0, vmax=5).format("{:.2f}"))

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    df_pca['cluster'] = df['cluster']

    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='cluster', palette='viridis', ax=ax4)
    ax4.set_title("PCA of Clusters")
    st.pyplot(fig4)

    # --- Decision Tree ---
    st.header("🌳 Decision Tree Classification")
    target_col = 'Niveau Maturité'

    if target_col in df.columns:
        features_dt = df.drop(columns=cols_to_exclude, errors='ignore')
        features_dt = features_dt.select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col].dropna()
        features_dt = features_dt.loc[y.index]

        X_train, X_test, y_train, y_test = train_test_split(features_dt, y, test_size=0.3, stratify=y, random_state=42)

        clf = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
        clf.fit(X_train, y_train)

        # --- Graphviz tree rendering ---
        st.subheader("🎯 Visualize Decision Tree")
        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=X_train.columns,
            class_names=[str(c) for c in clf.classes_],
            filled=True, rounded=True,
            special_characters=True
        )
        st.graphviz_chart(dot_data)

    else:
        st.warning("The column 'Niveau Maturité' was not found in the dataset.")

else:
    st.info("👈 Upload a file to begin.")
