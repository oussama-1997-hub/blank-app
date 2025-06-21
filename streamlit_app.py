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
st.sidebar.header("📂 Upload your Excel file")
file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

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
    df = pd.read_excel(file)
    st.success("✅ Excel file uploaded successfully!")
    st.dataframe(df.head())

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
    tabs = st.tabs(["📊 Clustering", "🧭 PCA", "📡 Radar", "🔥 Heatmaps", "🌳 Decision Tree", "⚙️ Application", "📥 Export"])

    # ----- Clustering Tab -----
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
            1: 'Niveau Initial',
            2: 'Niveau Intégré',
            0: 'Niveau Avancé'
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

    # ----- PCA Tab -----
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

    # ----- Radar Chart Tab -----
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

    # ----- Heatmaps Tab -----
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

    # ----- Decision Tree Tab -----
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

    # ----- Application Tab (nouveau) -----
    with tabs[5]:
        st.header("⚙️ Application : Évaluation & Feuille de Route Personnalisée")

        # Préparation des modèles à utiliser (KMeans et Decision Tree entraînés)
        # On reprend kmeans et clf déjà entraînés dans les tabs précédents :
        # Pour éviter erreurs, on retient final_k et clf entraînés dans la portée globale
        # Mais comme on a défini kmeans et clf dans les tabs précédents, on doit s'assurer qu'ils sont bien définis ici

        if 'kmeans' not in locals() or 'clf' not in locals():
            st.error("Veuillez d'abord exécuter les tabs Clustering et Decision Tree pour entraîner les modèles.")
            st.stop()

        # Affichage sélection d'entreprise à tester : par défaut entreprise 5 (index=4 si zero-based)
        st.markdown("### Sélection de l'entreprise à évaluer")
        entreprise_options = list(df.index)
        default_idx = 4 if len(df) > 4 else 0  # entreprise 5 = index 4
        entreprise_idx = st.selectbox("Choisissez une entreprise (index):", entreprise_options, index=default_idx)
        entreprise = df.loc[entreprise_idx]

        st.markdown("#### Scores de maturité sous-dimensions sélectionnées")
        entreprise_features = entreprise[selected_features].values.reshape(1, -1)
        st.dataframe(pd.DataFrame(entreprise_features, columns=selected_features))
        # --- 1. Prédiction cluster KMeans (niveau réel) ---
        entreprise_scaled = scaler.transform(entreprise[selected_features].values.reshape(1, -1))
        predicted_cluster = kmeans.predict(entreprise_scaled)[0]
        predicted_cluster_label = cluster_label_map.get(predicted_cluster, "Inconnu")

        st.write(f"**Niveau réel (KMeans cluster) prédit :** {predicted_cluster_label}")

        # --- 2. Prédiction arbre de décision (niveau prédit) ---
        # Préparer features DT (technos et lean dummies)
        # Suppose entreprise is a Series (like df.loc[5])
        features_dt_new = pd.DataFrame([entreprise]).drop(columns=exclude_cols, errors='ignore')
        features_dt_new.columns = [col.strip() for col in features_dt_new.columns]  # optional cleaning
        features_dt_new = features_dt_new.select_dtypes(include=[np.number]).fillna(0)
        features_dt_array = features_dt_new.values.reshape(1, -1)  # final input to model


        predicted_dt = clf.predict(features_dt_new)[0]
        st.write(f"**Niveau prédit (arbre de décision) :** {predicted_dt}")

        # --- 3. Analyse comparative & scénarios ---
        st.markdown("### 🔍 Analyse comparative & recommandations")

        # Mapping label ordre pour comparer niveaux
        label_order = {'Niveau Initial': 1, 'Niveau Intégré': 2, 'Niveau Avancé': 3}

        niveau_reel_ord = label_order.get(predicted_cluster_label, 0)
        niveau_pred_ord = label_order.get(predicted_dt, 0)

        if niveau_pred_ord < niveau_reel_ord:
            st.warning("⚠️ Scénario 1 : Niveau prédit < niveau réel (retard technologique)")
            st.write("• Renforcer l’adoption de technologies clés.")
            st.write("• Identifier les outils prioritaires via l’arbre de décision (nœuds parents).")
        elif niveau_pred_ord > niveau_reel_ord:
            st.warning("⚠️ Scénario 2 : Niveau prédit > niveau réel (retard maturité organisationnelle)")
            st.write("• Prioriser l’amélioration des sous-dimensions de maturité.")
            st.write("• Identifier les sous-dimensions présentant les plus grands écarts avec la moyenne du groupe cible.")
        else:
            st.success("✅ Scénario 3 : Niveau prédit = niveau réel (alignement)")
            st.write("• Améliorer en parallèle la maturité organisationnelle et l’adoption d’outils.")
            st.write("• Viser les outils du nœud parent dans l’arbre et les sous-dimensions où l’écart est le plus important.")
        # --- 3b. Radar Chart personnalisé : Entreprise vs Cluster cible ---
        cluster_means = df.groupby('cluster')[selected_features].mean()
        entreprise_scores = entreprise[selected_features]
        target_cluster = predicted_cluster
        next_cluster = target_cluster + 1 if target_cluster + 1 <= final_k else if (target_cluster=0 or target_cluster=3) target_cluster+1
        st.markdown("### 📡 Radar Chart : Entreprise vs Cluster Cible")
        try:    
            entreprise_scores_list = entreprise[selected_features].values.flatten().tolist()
            cluster_target_mean = cluster_means.loc[next_cluster][selected_features].values.tolist()
            fig_compare_radar = go.Figure()
            fig_compare_radar.add_trace(go.Scatterpolar(
                r=entreprise_scores_list,
                theta=selected_features,
                fill='toself',
                name="Entreprise",
                line=dict(color='rgba(255, 0, 0, 1)', width=3),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))

            fig_compare_radar.add_trace(go.Scatterpolar(
                r=cluster_target_mean,
                theta=selected_features,
                fill='toself',
                name="Moyenne du cluster cible",
                line=dict(color='rgba(0, 0, 139, 1)', width=3),
                fillcolor='rgba(0, 0, 139, 0.3)'
            ))

            fig_compare_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=True,
                height=600
            )

            st.plotly_chart(fig_compare_radar)

        except Exception as e:
            st.error(f"Erreur lors de la génération du Radar Chart personnalisé : {e}")
        # --- 4. Feuille de route personnalisée ---

        st.markdown("### 🗺️ Feuille de route personnalisée")

        # 4a. Calcul des gaps par sous-dimension (comparaison cluster cible vs entreprise)
        # Récupérer moyennes du cluster cible (niveau réel + 1 si possible)
       


        # Calcul des écarts entre l'entreprise et le cluster cible

        gaps = entreprise_scores - cluster_means.loc[next_cluster]
        
        # Ne garder que les écarts négatifs
        negative_gaps = gaps[gaps < 0]
        
        # Trier du plus grand écart négatif au plus petit (valeurs les plus éloignées)
        gaps_sorted = negative_gaps.sort_values()
        
        # Affichage
        st.subheader("🔻 Sous-dimensions avec un écart négatif (priorité d'amélioration)")
        
        # On affiche tous les écarts négatifs triés, sans limite
        gap_values = pd.to_numeric(gaps_sorted.values, errors='coerce')
        gap_df = pd.DataFrame({
            'Sous-dimension': gaps_sorted.index,
            'Écart': np.round(gap_values, 2)
        })
        
        st.table(gap_df)



        # 4b. Feuille de route technologique personnalisée
        st.subheader("Méthodes Lean & Technologies à adopter")

        # Définir colonnes Lean et Tech disponibles (dummy columns)
        lean_cols = [col for col in df.columns if col.startswith('Lean_')]
        tech_cols = [col for col in df.columns if col.startswith('Tech_')]

        # Moyennes cluster cible
        lean_cluster_mean = df.loc[df['cluster'] == next_cluster, lean_cols].mean()
        tech_cluster_mean = df.loc[df['cluster'] == next_cluster, tech_cols].mean()

        # Outils non adoptés par l'entreprise (valeur = 0)
        lean_to_adopt = lean_cluster_mean[(lean_cluster_mean > 0) & (entreprise[lean_cluster_mean.index] == 0)]
        tech_to_adopt = tech_cluster_mean[(tech_cluster_mean > 0) & (entreprise[tech_cluster_mean.index] == 0)]

        # Trier par taux d'adoption décroissant
        lean_to_adopt = lean_to_adopt.sort_values(ascending=False)
        tech_to_adopt = tech_to_adopt.sort_values(ascending=False)

        # Affichage méthodes Lean à adopter
        if not lean_to_adopt.empty:
            lean_df = pd.DataFrame({
                "Méthode Lean": lean_to_adopt.index.str.replace('Lean_', ''),
                "Taux d'adoption dans cluster cible": lean_to_adopt.values.round(2)
            })
            st.write("### Méthodes Lean à adopter en priorité")
            st.dataframe(lean_df)
        else:
            st.info("Aucune méthode Lean prioritaire à adopter.")

        # Affichage technologies Industrie 4.0 à adopter
        if not tech_to_adopt.empty:
            tech_df = pd.DataFrame({
                "Technologie Industrie 4.0": tech_to_adopt.index.str.replace('Tech_', ''),
                "Taux d'adoption dans cluster cible": tech_to_adopt.values.round(2)
            })
            st.write("### Technologies Industrie 4.0 à adopter en priorité")
            st.dataframe(tech_df)
        else:
            st.info("Aucune technologie prioritaire à adopter.")

    # ----- Export Tab -----
    with tabs[6]:
        st.header("📥 Export des données enrichies")
        if st.button("Télécharger le fichier CSV avec clusters et maturité"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Télécharger CSV", data=csv, file_name="df_with_clusters.csv", mime="text/csv")

else:
    st.info("⏳ Veuillez uploader un fichier CSV pour commencer.")
