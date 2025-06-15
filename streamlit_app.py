# streamlit_kmeans_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner display
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="KMeans Clustering App", layout="wide")
st.title("🔍 KMeans Clustering on Survey Data")

# --- Sidebar inputs ---
st.sidebar.header("🔧 Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your cleaned CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    colonnes = st.sidebar.multiselect("Select Likert-scale columns for clustering", df.columns.tolist())
    if not colonnes:
        st.warning("Please select at least one column to proceed.")
        st.stop()

    # Fill missing values
    X = df[colonnes].copy()
    X.fillna(X.mean(), inplace=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose number of clusters
    k_min, k_max = 2, 10
    st.sidebar.subheader("Number of Clusters")
    k_range = st.sidebar.slider("Select range of K", min_value=2, max_value=10, value=(2, 6))

    inertia = []
    silhouette_scores = []

    for k in range(k_range[0], k_range[1]+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # Elbow plot
    st.subheader("📉 Elbow Method")
    fig_elbow, ax = plt.subplots()
    ax.plot(range(k_range[0], k_range[1]+1), inertia, marker='o')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig_elbow)

    # Silhouette plot
    st.subheader("📈 Silhouette Scores")
    fig_sil, ax2 = plt.subplots()
    ax2.plot(range(k_range[0]+1, k_range[1]+1), silhouette_scores, marker='o')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores')
    st.pyplot(fig_sil)

    # Cluster assignment
    optimal_k = st.sidebar.number_input("Select K for final clustering", min_value=2, max_value=10, value=3)

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)
    df['Cluster'] = clusters

    st.subheader("📊 Cluster Sizes")
    st.write(df['Cluster'].value_counts())

    st.subheader("📋 Average Survey Scores per Cluster")
    st.dataframe(df.groupby('Cluster')[colonnes].mean().T.style.background_gradient(cmap='YlGnBu'))

    # PCA visualization
    st.subheader("🧬 PCA Cluster Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig_pca, ax3 = plt.subplots()
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
    ax3.set_xlabel("PCA Component 1")
    ax3.set_ylabel("PCA Component 2")
    ax3.set_title("Clusters (PCA Projection)")
    legend = ax3.legend(*scatter.legend_elements(), title="Cluster")
    st.pyplot(fig_pca)

else:
    st.info("Upload a CSV file to begin.")
