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
import plotly.graph_objects as go
import graphviz

st.set_page_config(page_title="Lean 4.0 Cluster & Tree App", layout="wide")
st.title("🔍 Lean 4.0 Clustering & Decision Tree Dashboard")

# --- Sidebar Config ---
st.sidebar.header("📂 Upload your CSV file")
file = st.sidebar.file_uploader("Upload df_cleaned_with_dummies.csv", type="csv")

# --- Columns of Interest ---
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

exclude_cols = ['Indicateurs suivis', 'Zone investissement principale', 'Typologie de production',
                'Type de flux', 'Pays ', 'Méthodes Lean ', 'Technologies industrie 4.0',
                'cluster', 'Cluster', 'Feature_Cluster', 'Niveau Maturité', 'Cluster Label'] + colonnes

if file:
    df = pd.read_csv(file)
    st.success("✅ File loaded successfully")

    # --- Feature selection ---
    selected_features = st.sidebar.multiselect(
        "📌 Select features for clustering", colonnes, default=colonnes
    )

    features = df[selected_features].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Clustering to generate labels if needed ---
    # --- Ensure 'Niveau Maturité' column exists and is valid ---
    if 'Niveau Maturité' not in df.columns or df['Niveau Maturité'].isnull().all():
        st.warning("ℹ️ 'Niveau Maturité' not found or is empty. Running KMeans to assign clusters.")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        df['Niveau Maturité'] = df['cluster']
    
    # Ensure it's integer
    df['Niveau Maturité'] = df['Niveau Maturité'].astype(int)
    
    # Filter unexpected values
    df = df[df['Niveau Maturité'].isin([0, 1, 2])]
    
    # Map numeric to label
    maturity_label_map = {1: 'Niveau Initial', 2: 'Niveau Intégré', 0: 'Niveau Avancé'}
    maturity_order = ['Niveau Initial', 'Niveau Intégré', 'Niveau Avancé']
    df['Niveau Maturité Label'] = df['Niveau Maturité'].map(maturity_label_map)


    # Map numeric to labels
    maturity_label_map = {1: 'Niveau Initial', 2: 'Niveau Intégré', 0: 'Niveau Avancé'}
    maturity_order = ['Niveau Initial', 'Niveau Intégré', 'Niveau Avancé']
    df['Niveau Maturité Label'] = df['Niveau Maturité'].map(maturity_label_map)

    # --- Tabs ---
    tabs = st.tabs(["📊 Clustering", "🧭 PCA Visualization", "🔥 Heatmap", "🌳 Decision Tree", "📥 Export"])

    with tabs[0]:
        st.header("📋 Cluster Summary by Maturity Level")
        summary_df = df['Niveau Maturité Label'].value_counts().reindex(maturity_order).reset_index()
        summary_df.columns = ['Niveau de maturité Lean 4.0', "Nombre d'entreprises"]
        st.table(summary_df)

        st.subheader("📡 Cluster Profile Radar Chart")
        cluster_avg = df.groupby('Niveau Maturité Label')[selected_features].mean().reindex(maturity_order)
        categories = selected_features
        fig_radar = go.Figure()
        for label in cluster_avg.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_avg.loc[label].values,
                theta=categories,
                fill='toself',
                name=label
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig_radar)

    with tabs[1]:
        st.header("🧭 PCA Cluster Visualization")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
        df_pca['Niveau Maturité Label'] = df['Niveau Maturité Label']

        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Niveau Maturité Label',
                        palette='viridis', hue_order=maturity_order, ax=ax3)
        ax3.set_title("PCA of Lean 4.0 Maturity Levels")
        st.pyplot(fig3)

    with tabs[2]:
        st.header("🔥 Heatmap of Average Scores by Maturity Level")
        avg_scores = df.groupby('Niveau Maturité Label')[selected_features].mean().reindex(maturity_order)

        fig, ax = plt.subplots(figsize=(14, max(9, len(selected_features) * 0.5)))
        sns.heatmap(avg_scores.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.8, ax=ax)
        ax.set_title("Average Scores per Maturity Level", fontsize=16)
        st.pyplot(fig)

    with tabs[3]:
        st.header("🌳 Decision Tree Classification")
        target_col = 'Niveau Maturité Label'

        features_dt = df.drop(columns=exclude_cols, errors='ignore')
        features_dt = features_dt.select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col].dropna()
        features_dt = features_dt.loc[y.index]

        st.write("Sample counts per maturity level:", y.value_counts())

        if y.nunique() > 1 and len(y) > 5:
            max_depth = st.slider("Max Depth", 1, 10, 4)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 4)

            X_train, X_test, y_train, y_test = train_test_split(
                features_dt, y, test_size=0.3, stratify=y, random_state=42)

            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            clf.fit(X_train, y_train)

            importances = pd.Series(clf.feature_importances_, index=X_train.columns)
            top_importances = importances[importances > 0].sort_values(ascending=False).head(20)

            st.subheader("🔎 Feature Importances")
            if not top_importances.empty:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                top_importances.plot(kind='barh', ax=ax5, color='steelblue')
                ax5.set_title("Top Feature Importances")
                st.pyplot(fig5)
            else:
                st.info("No features with importance found.")

            st.subheader("🎯 Decision Tree Visualization")
            dot_data = export_graphviz(clf, out_file=None, feature_names=X_train.columns,
                                       class_names=clf.classes_, filled=True, rounded=True, special_characters=True)
            st.graphviz_chart(dot_data)

        else:
            st.warning("⚠️ Not enough data or class variety to train the Decision Tree. Please check your input.")

    with tabs[4]:
        st.header("📥 Export Results")
        st.download_button("Download full dataset", data=df.to_csv(index=False),
                           file_name="lean4_maturity_clustered.csv", mime="text/csv")

else:
    st.info("👈 Please upload a CSV file to begin.")
