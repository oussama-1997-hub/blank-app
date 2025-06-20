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

# --- Dimension to sub-dimensions mapping ---
dimension_map = {
    "Leadership": [
        "Leadership - Engagement Lean ",
        "Leadership - Engagement DT",
        "Leadership - Stratégie ",
        "Leadership - Communication"
    ],
    "Supply Chain": [
        "Supply Chain - Collaboration inter-organisationnelle",
        "Supply Chain - Traçabilité",
        "Supply Chain - Impact sur les employées"
    ],
    "Opérations": [
        "Opérations - Standardisation des processus",
        "Opérations - Juste-à-temps (JAT)",
        "Opérations - Gestion des résistances"
    ],
    "Technologies": [
        "Technologies - Connectivité et gestion des données",
        "Technologies - Automatisation",
        "Technologies - Pilotage du changement"
    ],
    "Organisation Apprenante": [
        "Organisation apprenante  - Formation et développement des compétences",
        "Organisation apprenante  - Collaboration et Partage des Connaissances",
        "Organisation apprenante  - Flexibilité organisationnelle"
    ]
}

exclude_cols = ['Indicateurs suivis', 'Zone investissement principale', 'Typologie de production',
                'Type de flux', 'Pays ', 'Méthodes Lean ', 'Technologies industrie 4.0',
                'cluster', 'Cluster', 'Feature_Cluster', 'Niveau Maturité', 'Cluster Label'] + sum(dimension_map.values(), [])

if file:
    df = pd.read_csv(file)
    st.success("✅ File loaded successfully")

    # --- Sidebar: Select sub-dimensions grouped by dimension ---
    st.sidebar.markdown("### 📌 Sélectionner les sous-dimensions par dimension")
    selected_features = []
    for dimension, sub_dims in dimension_map.items():
        with st.sidebar.expander(f"🧩 {dimension}"):
            selected = st.multiselect(f"Sous-dimensions de {dimension}", sub_dims, default=sub_dims, key=dimension)
            selected_features.extend(selected)

    if not selected_features:
        st.sidebar.warning("⚠️ Veuillez sélectionner au moins une sous-dimension.")
        st.stop()

    # --- Sidebar: Select dimensions to show on Radar Chart ---
    st.sidebar.markdown("### 🎯 Choisissez les dimensions à afficher dans le Radar Chart")
    all_dimensions = list(dimension_map.keys())
    selected_dimensions_for_radar = st.sidebar.multiselect(
        "Dimensions pour Radar Chart",
        options=all_dimensions,
        default=all_dimensions,
        key='radar_dimensions'
    )

    # Build feature list for radar chart based on selected dimensions
    selected_features_for_radar = []
    for dim in selected_dimensions_for_radar:
        selected_features_for_radar.extend(dimension_map[dim])

    # --- Prepare features for clustering ---
    features = df[selected_features].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Tabs for visualization ---
    tabs = st.tabs(["📊 Clustering", "🧭 PCA", "📡 Radar", "🔥 Heatmaps", "🌳 Decision Tree", "📥 Export"])

    with tabs[0]:
        st.header("📊 KMeans Clustering")
        k_range = st.slider("Select K range", 2, 10, (2, 6))

        inertia, silhouette_scores = [], []
        for k in range(k_range[0], k_range[1] + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(scaled_features)
            inertia.append(model.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, model.labels_))

        st.subheader("Elbow Method")
        fig1, ax1 = plt.subplots()
        ax1.plot(range(k_range[0], k_range[1] + 1), inertia, marker='o')
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("K")
        ax1.set_ylabel("Inertia")
        st.pyplot(fig1)

        st.subheader("Silhouette Score")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(k_range[0], k_range[1] + 1), silhouette_scores, marker='o')
        ax2.set_title("Silhouette Scores")
        ax2.set_xlabel("K")
        ax2.set_ylabel("Score")
        st.pyplot(fig2)

        final_k = st.selectbox("Select final K", list(range(k_range[0], k_range[1] + 1)))
        kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_features)

        # Maturity label map
        cluster_label_map = {
            0: 'Niveau Avancé',
            1: 'Niveau Initial',
            2: 'Niveau Intégré'
        }
        df['Niveau de maturité Lean 4.0'] = df['cluster'].map(cluster_label_map)

        st.subheader("📋 Cluster Summary")
        cluster_counts = df['cluster'].value_counts().sort_index()
        summary_df = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Nombre d\'entreprises': cluster_counts.values,
            'Niveau de maturité Lean 4.0': cluster_counts.index.map(cluster_label_map)
        })
        st.table(summary_df)

    with tabs[1]:
        st.header("🧭 PCA Cluster Visualization")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
        df_pca['label'] = df['Niveau de maturité Lean 4.0']

        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='label', palette='Set2', ax=ax3)
        ax3.set_title("PCA of Clusters")
        st.pyplot(fig3)

    with tabs[2]:
        st.header("📡 Radar Chart - Profils par Dimension")
        try:
            cluster_avg = df.groupby('Niveau de maturité Lean 4.0')[selected_features_for_radar].mean().dropna(axis=1, how='any')
            available_features = cluster_avg.columns.tolist()

            custom_colors = {
                'Niveau Initial': {
                    'line': 'rgba(0, 0, 139, 1)',       # Dark Blue opaque
                    'fill': 'rgba(0, 0, 139, 0.5)'      # Dark Blue transparent
                },
                'Niveau Avancé': {
                    'line': 'rgba(173, 216, 230, 1)',   # Light Blue opaque
                    'fill': 'rgba(173, 216, 230, 0.3)'  # Light Blue transparent
                },
                'Niveau Intégré': {
                    'line': 'rgba(255, 0, 0, 1)',       # Red opaque
                    'fill': 'rgba(255, 0, 0, 0.3)'      # Red transparent
                }
            }

            if cluster_avg.empty:
                st.warning("Pas de données disponibles pour le radar. Veuillez vérifier la sélection.")
            else:
                fig_radar = go.Figure()
                for label in cluster_avg.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=cluster_avg.loc[label].values,
                        theta=available_features,
                        fill='toself',
                        name=label,
                        line=dict(color=custom_colors[label]['line'], width=3),
                        fillcolor=custom_colors[label]['fill']
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=True,
                    height=600
                )
                st.plotly_chart(fig_radar)

        except Exception as e:
            st.error(f"Erreur du Radar Chart : {e}")

    with tabs[3]:
        st.header("🔥 Heatmaps of Average Scores, Lean Methods & Industry 4.0 Tech")

        # Average survey scores heatmap (selected_features)
        avg_scores = df.groupby('Niveau de maturité Lean 4.0')[selected_features].mean()

        # Detect Lean and Tech dummy columns
        tech_cols = [col for col in df.columns if col.startswith('Tech_')]
        lean_cols = [col for col in df.columns if col.startswith('Lean_')]

        lean_avg = df.groupby('Niveau de maturité Lean 4.0')[lean_cols].mean() if lean_cols else pd.DataFrame()
        tech_avg = df.groupby('Niveau de maturité Lean 4.0')[tech_cols].mean() if tech_cols else pd.DataFrame()

        fig, axs = plt.subplots(3, 1, figsize=(16, 18))

        sns.heatmap(avg_scores.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.8, ax=axs[0])
        axs[0].set_title("Average Survey Scores by Maturity Level", fontsize=16)

        if lean_avg.empty:
            axs[1].text(0.5, 0.5, "No Lean methods columns detected.", ha='center', va='center', fontsize=14)
            axs[1].axis('off')
        else:
            sns.heatmap(lean_avg.T, cmap="Oranges", annot=True, fmt=".2f", linewidths=0.8, ax=axs[1])
            axs[1].set_title("Average Lean Methods Usage by Maturity Level", fontsize=16)

        if tech_avg.empty:
            axs[2].text(0.5, 0.5, "No Industry 4.0 tech columns detected.", ha='center', va='center', fontsize=14)
            axs[2].axis('off')
        else:
            sns.heatmap(tech_avg.T, cmap="PuRd", annot=True, fmt=".2f", linewidths=0.8, ax=axs[2])
            axs[2].set_title("Average Industry 4.0 Technologies Usage by Maturity Level", fontsize=16)

        plt.tight_layout()
        st.pyplot(fig)

    with tabs[4]:
        st.header("🌳 Decision Tree Classification")
        target_col = 'Niveau de maturité Lean 4.0'

        if target_col in df.columns:
            features_dt = df.drop(columns=exclude_cols, errors='ignore')
            features_dt = features_dt.select_dtypes(include=[np.number]).fillna(0)
            y = df[target_col].dropna()
            features_dt = features_dt.loc[y.index]

            max_depth = st.slider("Max Depth", 1, 10, 4)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 4)

            X_train, X_test, y_train, y_test = train_test_split(features_dt, y, test_size=0.3, stratify=y, random_state=42)
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
                                       class_names=[str(c) for c in clf.classes_],
                                       filled=True, rounded=True, special_characters=True)
            st.graphviz_chart(dot_data)

        else:
            st.warning("🛑 'Niveau de maturité Lean 4.0' not found in dataset.")

    with tabs[5]:
        # This will be the next module in your Streamlit app - called "🚀 Application Personnalisée"
# It will allow evaluating a single company (from dataset or user input)
# and generate maturity prediction + roadmaps (maturity + technological)

# --- Inside your app, after the last tab "Export" ---
with tabs[5]:
    st.header("🚀 Application personnalisée pour une entreprise")

    st.markdown("""
    Cette section permet de :
    - Prédire le niveau de maturité Lean 4.0 d'une entreprise
    - Détecter écarts entre outils utilisés et maturité
    - Générer des feuilles de route personnalisées (outils et dimensions)
    """)

    # --- Load company data ---
    source_option = st.radio("Source des données de l'entreprise :", ["Exemple Entreprise 5", "Depuis le dataset", "Saisie manuelle"])

    if source_option == "Exemple Entreprise 5":
        example_row = {
            'Lean_QRQC': 1, 'Lean_5S': 1, 'Lean_Value Stream Mapping (VSM)': 1,
            'Lean_TPM / TRS method': 1, 'Lean_Takt Time': 1,
            'Tech_Intelligence Artificielle': 1, 'Tech_ERP': 1,
            # Technologies/méthodes non utilisées seront 0 automatiquement
        }
        example_scores = {
            'Leadership - Engagement Lean ': 4.0,
            'Leadership - Engagement DT': 2.0,
            'Leadership - Stratagie ': 2.0,
            'Leadership - Communication': 3.0,
            'Supply Chain - Collaboration inter-organisationnelle': 3.0,
            'Supply Chain - Tracabilite': 2.0,
            'Supply Chain - Impact sur les employees': 3.0,
            'Opérations - Standardisation des processus': 2.0,
            'Opérations - Juste-à-temps (JAT)': 3.0,
            'Opérations - Gestion des résistances': 2.0,
            'Technologies - Connectivité et gestion des données': 3.0,
            'Technologies - Automatisation': 2.0,
            'Technologies - Pilotage du changement': 3.0,
            'Organisation apprenante  - Formation et développement des compétences': 3.0,
            'Organisation apprenante  - Collaboration et Partage des Connaissances': 3.0,
            'Organisation apprenante  - Flexibilité organisationnelle': 3.0
        }

        # Create DataFrame row
        lean_tech_cols = [col for col in df.columns if col.startswith("Lean_") or col.startswith("Tech_")]
        row_data = {col: example_row.get(col, 0) for col in lean_tech_cols}
        subdim_cols = selected_features
        row_data.update({col: example_scores.get(col, 0) for col in subdim_cols})
        row_df = pd.DataFrame([row_data])

        # --- Scale scores and predict cluster (maturity real) ---
        scaled_input = scaler.transform(row_df[subdim_cols])
        predicted_cluster = kmeans.predict(scaled_input)[0]
        maturity_label = cluster_label_map.get(predicted_cluster, "Inconnu")

        # --- Decision tree prediction (based on lean/tech) ---
        tech_lean_input = row_df[lean_tech_cols].reindex(columns=X_train.columns, fill_value=0)
        predicted_tree_label = clf.predict(tech_lean_input)[0]

        st.subheader(f"📌 Niveau réel (par clustering) : **{maturity_label}**")
        st.subheader(f"🌳 Niveau prédit (arbre de décision) : **{predicted_tree_label}**")

        # --- Analyse des écarts ---
        st.markdown("### 🌀 Analyse comparative des niveaux")
        if maturity_label == predicted_tree_label:
            scenario = "Scénario 3 : Aligné"
            reco = "Améliorer en parallèle la maturité organisationnelle et l'adoption d'outils."
        elif maturity_label == "Niveau Initial" and predicted_tree_label in ["Niveau Intégré", "Niveau Avancé"]:
            scenario = "Scénario 2 : Outils avancés mais organisation faible"
            reco = "Prioriser l'amélioration des sous-dimensions de maturité."
        else:
            scenario = "Scénario 1 : Retard technologique"
            reco = "Renforcer l'adoption de technologies clés."

        st.info(f"**{scenario}**\n\n✍️ **Recommandation principale** : {reco}")

        # --- Roadmap sous-dimensions ---
        st.markdown("### 🚀 Feuille de route d'amélioration de maturité")
        avg_cluster = df[df['Niveau de maturité Lean 4.0'] == 'Niveau Intégré'][subdim_cols].mean()
        gaps = avg_cluster - row_df[subdim_cols].iloc[0]
        gap_df = pd.DataFrame({
            'Sous-dimension': gaps.index,
            'Score Entreprise': row_df[subdim_cols].iloc[0].values,
            'Moyenne Cluster 2': avg_cluster.values,
            'Ecart': gaps.values
        }).sort_values(by='Ecart')

        top_gaps = gap_df.nsmallest(5, 'Ecart')
        st.dataframe(top_gaps)

        # --- Roadmap outils ---
        st.markdown("### 📆 Feuille de route technologique personnalisée")
        cluster_df = df[df['Niveau de maturité Lean 4.0'] == 'Niveau Intégré']
        adoption_rates = cluster_df[lean_tech_cols].mean()
        not_used = row_df[lean_tech_cols].iloc[0] == 0
        to_adopt = adoption_rates[not_used].sort_values(ascending=False)

        tools_df = pd.DataFrame({
            'Outil / Technologie': to_adopt.index,
            'Taux d'adoption Cluster 2': to_adopt.values,
            'Priorité': pd.cut(to_adopt.values, bins=[0, 0.2, 0.5, 1], labels=['Faible', 'Moyenne', 'Haute'])
        })

        st.dataframe(tools_df)

    else:
        st.warning("Cette version préliminaire n'inclut que l'exemple. Les modes Dataset et Saisie manuelle seront ajoutés après validation.")


else:
    st.info("👈 Please upload a CSV file to begin.")
