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
    cluster_counts = df['cluster'].value_counts().sort_index()
    niveau_maturite_map = {0: 'Niveau Avancé', 1: 'Niveau Initial', 2: 'Niveau Intégré'}
    cluster_summary = pd.DataFrame({
        'Cluster': cluster_counts.index,
        'Nombre d\'entreprises': cluster_counts.values,
        'Niveau de maturité Lean 4.0': cluster_counts.index.map(niveau_maturite_map)
    })
    st.table(cluster_summary)

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    df_pca['cluster'] = df['cluster']

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='cluster', palette='viridis', ax=ax3)
    ax3.set_title("PCA of Clusters")
    st.pyplot(fig3)

    # --- Heatmaps ---
    avg_scores = df.groupby('cluster')[colonnes].mean()
    st.subheader("📈 Average Survey Scores per Cluster (Heatmap)")
    fig, ax = plt.subplots(figsize=(14, max(9, len(colonnes)*0.5)))
    sns.heatmap(avg_scores.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.8, linecolor='gray',
                cbar_kws={'label': 'Average Score', 'shrink': 0.75, 'aspect': 15, 'pad': 0.02}, ax=ax)
    ax.set_title("Average Survey Scores per Cluster", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Cluster", fontsize=14, labelpad=15)
    ax.set_ylabel("Survey Features", fontsize=14, labelpad=15)
    ax.set_yticks(np.arange(len(colonnes)) + 0.5)
    ax.set_yticklabels(colonnes, fontsize=12, rotation=0)
    ax.tick_params(axis='x', labelsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

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

        st.subheader("🔎 Top Feature Importances (non-zero only)")
        importances = pd.Series(clf.feature_importances_, index=X_train.columns)
        non_zero_importances = importances[importances > 0].sort_values(ascending=False).head(20)
        if not non_zero_importances.empty:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            non_zero_importances.plot(kind='barh', ax=ax5, color='steelblue')
            ax5.set_title("Top Non-Zero Feature Importances")
            ax5.set_xlabel("Importance")
            ax5.set_ylabel("Feature")
            plt.tight_layout()
            st.pyplot(fig5)
        else:
            st.info("ℹ️ No features with non-zero importance were found.")

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

        # --- Apply Model to New Company ---
        st.header("🏭 Personalized Maturity Assessment")
        st.markdown("Provide responses below to simulate maturity evaluation for a new company.")

        st.subheader("📋 Sous-dimensions (1-5 scale)")
        new_scores = [st.slider(label, 1, 5, 3) for label in colonnes]
        input_features = pd.DataFrame([new_scores], columns=colonnes)
        scaled_input = scaler.transform(input_features)
        predicted_cluster = kmeans.predict(scaled_input)[0]

        st.write(f"**KMeans assigned cluster**: {predicted_cluster} → Niveau estimé: {niveau_maturite_map.get(predicted_cluster, 'Inconnu')}")

        st.subheader("🛠 Méthodes Lean & Outils Industrie 4.0")
        lean_cols = [col for col in df.columns if 'lean_' in col.lower() or '5s' in col.lower()]
        tech_cols = [col for col in df.columns if 'tech' in col.lower() or 'iot' in col.lower() or 'cloud' in col.lower()]

        selected_methods = st.multiselect("Sélectionnez les méthodes Lean utilisées", lean_cols)
        selected_techs = st.multiselect("Sélectionnez les technologies Industrie 4.0 utilisées", tech_cols)

        input_binary = pd.DataFrame(np.zeros((1, len(X_train.columns))), columns=X_train.columns)
        for m in selected_methods + selected_techs:
            if m in input_binary.columns:
                input_binary.at[0, m] = 1

        predicted_tree = clf.predict(input_binary)[0]
        st.write(f"**Decision Tree predicted level**: {predicted_tree}")

        st.subheader("📌 Recommandations")
        if predicted_tree != niveau_maturite_map.get(predicted_cluster, ''):
            st.markdown("**➡️ Incohérence détectée entre niveau prédit et niveau réel.**")
            st.markdown("Adaptez vos outils ou renforcez les pratiques organisationnelles en conséquence.")
        else:
            st.markdown("✅ Votre profil est cohérent entre outils utilisés et maturité perçue.")

    else:
        st.warning("The column 'Niveau Maturité' was not found in the dataset.")

else:
    st.info("👈 Upload a file to begin.")
