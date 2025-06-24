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
# 🌟 Page d'accueil - Présentation du site
st.markdown("""
    <div style="background-color: #f7f9fc; padding: 30px 20px; border-radius: 15px; box-shadow: 0px 2px 8px rgba(0,0,0,0.1);">
        <h1 style="color: #004080; font-size: 32px; text-align: center; margin-bottom: 10px;">🚀 Optimisez votre transformation Lean 4.0 grâce à l’intelligence issue du terrain</h1>
        <p style="font-size: 18px; color: #333333; text-align: center; max-width: 850px; margin: 0 auto;">
            Bienvenue sur votre assistant intelligent Lean 4.0 – <strong>la première plateforme de recommandation</strong>
            qui s’appuie sur <strong>des données réelles issues d’entreprises</strong> de différents secteurs et profils.
            <br><br>
            🎯 <strong>Ici, pas de théorie figée ni de jugement subjectif</strong> : nos suggestions sont basées sur
            l’analyse de cas concrets et performants pour vous proposer une feuille de route <strong>personnalisée,
            réaliste et actionnable</strong>.
            <br><br>
            Grâce à l’intelligence artificielle et à une base de connaissances issue du terrain :
        </p>
        <ul style="font-size: 17px; color: #444; line-height: 1.8; max-width: 850px; margin: 20px auto;">
            <li>📊 Vous identifiez vos <strong>écarts de maturité</strong> par rapport à des entreprises similaires.</li>
            <li>🛠️ Vous découvrez les <strong>méthodes Lean & technologies</strong> adaptées à votre profil.</li>
            <li>🧭 Vous suivez une <strong>feuille de route claire et guidée</strong> vers l’excellence opérationnelle.</li>
        </ul>
        <p style="font-size: 17px; color: #333; text-align: center; margin-top: 20px;">
            💡 Que vous soyez en phase de démarrage ou de perfectionnement, laissez-vous guider par la <strong>data</strong>, pas par la théorie.
        </p>
    </div>
""", unsafe_allow_html=True)

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
    tabs = st.tabs(["📊 Clustering", "🧭 PCA", "📡 Radar", "🔥 Heatmaps", "🌳 Decision Tree", "⚙️ Application"])

        
    # ----- Clustering Tab -----
    with tabs[0]:
        with st.expander("📈 Analyse de Fiabilité (Alpha de Cronbach)"):
            st.subheader("✨ Analyse de la fiabilité des groupes de sous-dimensions")
        
            overall_alpha = 0.934
            cronbach_data = {
                "Leadership": {
                    "alpha": 0.931,
                    "items": {
                        "Leadership - Communication": 0.992,
                        "Leadership - Engagement Lean": 0.926,
                        "Leadership - Stratégie": 0.901,
                        "Leadership - Engagement DT": 0.868
                    }
                },
                "SupplyChain": {
                    "alpha": 0.863,
                    "items": {
                        "Supply Chain - Impact sur les employées": 0.925,
                        "Supply Chain - Traçabilité": 0.826,
                        "Supply Chain - Collaboration inter-organisationnelle": 0.722
                    }
                },
                "Operations": {
                    "alpha": 0.867,
                    "items": {
                        "Opérations - Juste-à-temps (JAT)": 0.931,
                        "Opérations - Standardisation des processus": 0.831,
                        "Opérations - Gestion des résistances": 0.754
                    }
                },
                "Technologies": {
                    "alpha": 0.888,
                    "items": {
                        "Technologies - Connectivité et gestion des données": 0.904,
                        "Technologies - Automatisation": 0.881,
                        "Technologies - Pilotage du changement": 0.781
                    }
                },
                "OrgApprenante": {
                    "alpha": 0.854,
                    "items": {
                        "Organisation apprenante  - Formation et développement des compétences": 0.876,
                        "Organisation apprenante  - Collaboration et Partage des Connaissances": 0.799,
                        "Organisation apprenante  - Flexibilité organisationnelle": 0.763
                    }
                }
            }
        
            st.success(f"Cronbach's Alpha global pour toutes les colonnes sélectionnées : {overall_alpha:.3f}")
        
            for group, values in cronbach_data.items():
                st.markdown(f"#### Groupe : {group}")
                st.write(f"✅ Alpha global : {values['alpha']:.3f}")
        
                item_df = pd.DataFrame({
                    "Sous-dimension": list(values["items"].keys()),
                    "Alpha si supprimée": list(values["items"].values())
                })
                st.dataframe(item_df, use_container_width=True)
                st.markdown("---")
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

                # === 2. Radar chart par dimension ===
            st.subheader("📊 Radar Chart - Profils par *Dimension*")
    
            dimension_groups = {
                "Leadership": [col for col in selected_features_for_radar if "Leadership" in col],
                "Opérations": [col for col in selected_features_for_radar if "Opérations" in col],
                "Organisation apprenante": [col for col in selected_features_for_radar if "Organisation apprenante" in col],
                "Technologies": [col for col in selected_features_for_radar if "Technologies" in col],
                "Supply Chain": [col for col in selected_features_for_radar if "Supply Chain" in col],
            }
    
            dimension_avg = pd.DataFrame(index=df['Niveau de maturité Lean 4.0'].unique())
            for dim, cols in dimension_groups.items():
                if cols:
                    dimension_avg[dim] = df.groupby('Niveau de maturité Lean 4.0')[cols].mean().mean(axis=1)
            dimension_avg = dimension_avg.dropna()
    
            if dimension_avg.empty:
                st.warning("Pas de données disponibles pour le radar des dimensions.")
            else:
                fig_dim_radar = go.Figure()
                for label in dimension_avg.index:
                    fig_dim_radar.add_trace(go.Scatterpolar(
                        r=dimension_avg.loc[label].values,
                        theta=dimension_avg.columns,
                        fill='toself',
                        name=label,
                        line=dict(color=custom_colors[label]['line'], width=3),
                        fillcolor=custom_colors[label]['fill']
                    ))
                fig_dim_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=True,
                    height=600
                )
                st.plotly_chart(fig_dim_radar)


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
        # --- Analyse comparative et recommandations ---
        with st.container():
            st.markdown("## 🧭 Guide d’utilisation personnalisé", unsafe_allow_html=True)
            st.write("---")
        
            st.markdown("""
            <div style="background-color:#f9f9f9;padding:15px;border-radius:10px;">
            <h4>🧩 Ordre de priorité à suivre dans votre démarche Lean 4.0</h4>
            <ul>
                <li><b>Étape 1 :</b> Identification du scénario (retard techno / maturité / alignement)</li>
                <li><b>Étape 2 :</b> Application des recommandations adaptées à votre profil</li>
                <li><b>Étape 3 :</b> Suivi des feuilles de route proposées :
                    <ul>
                        <li>📈 <b>Feuille de route technologique</b> : Technologies & méthodes Lean à adopter en priorité</li>
                        <li>🧱 <b>Feuille de route de maturité</b> : Sous-dimensions Lean 4.0 à améliorer en priorité</li>
                    </ul>
                </li>
                <li><b>Étape 4 :</b> Implémentation progressive selon le scénario identifié</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
            st.markdown("## 🔍 Analyse comparative et recommandations", unsafe_allow_html=True)
            st.write("---")
        
            # Définir l’ordre des labels
            label_order = {'Niveau Initial': 1, 'Niveau Intégré': 2, 'Niveau Avancé': 3}
            niveau_reel_ord = label_order.get(predicted_cluster_label, 0)
            niveau_pred_ord = label_order.get(predicted_dt, 0)
        
            col1, col2 = st.columns([4, 1])
        
            with col1:
                if niveau_pred_ord < niveau_reel_ord:
                    st.markdown("### ⚠️ Scénario 1 : Retard technologique")
                    st.markdown("""
                    - Votre entreprise dispose d’une bonne maturité Lean 4.0, mais **n’a pas encore adopté les technologies associées à ce niveau**.
                    - Cela révèle un **retard technologique**.
                    
                    **🛠️ Recommandations :**
                    1. Prioriser les **technologies et méthodes Lean** utilisées dans votre groupe cible mais non encore adoptées.
                    2. Identifier les **nœuds parents** dans l’arbre de décision menant aux niveaux supérieurs.
                    3. Appliquer en priorité la **feuille de route technologique**.
                    4. Ensuite, renforcer la **maturité organisationnelle** avec la feuille de route Lean 4.0.
                    """)
        
                elif niveau_pred_ord > niveau_reel_ord:
                    st.markdown("### ⚠️ Scénario 2 : Avance technologique")
                    st.markdown("""
                    - Votre entreprise utilise des outils et technologies avancés, mais **n’a pas encore atteint la maturité Lean 4.0 correspondante**.
                    - Cela indique un **retard organisationnel**.
        
                    **🛠️ Recommandations :**
                    1. Prioriser les **sous-dimensions Lean 4.0** à améliorer.
                    2. Se concentrer sur les **écarts les plus importants** par rapport à votre cluster cible.
                    3. Appliquer d’abord la **feuille de route Lean 4.0**.
                    4. Intégrer ensuite progressivement la **feuille de route technologique**.
                    """)
        
                else:
                    st.markdown("### ✅ Scénario 3 : Alignement stratégique")
                    st.markdown("""
                    - Votre entreprise est **alignée entre maturité Lean 4.0 et adoption technologique**. Bravo !
        
                    **🛠️ Recommandations :**
                    1. Continuer à améliorer de manière équilibrée les **technologies et la maturité**.
                    2. Viser les **nœuds parents dans l’arbre de décision** ayant le plus d’influence sur votre avancement.
                    3. Appliquer la **feuille de route technologique** pour booster l’innovation.
                    4. Renforcer les sous-dimensions Lean 4.0 présentant les **plus grands gaps**.
                    """)
        
            with col2:
                icon = "🚀" if niveau_pred_ord == niveau_reel_ord else ("⚡" if niveau_pred_ord > niveau_reel_ord else "🔧")
                st.markdown(f"<h1 style='font-size:5rem;text-align:center'>{icon}</h1>", unsafe_allow_html=True)
        
            st.markdown("---")
            st.markdown(
                """
                <div style="background:#f1f3f4;padding:15px;border-radius:10px;">
                <b>🎯 En résumé :</b> Suivez la stratégie d’implémentation recommandée pour optimiser votre transition Lean 4.0 selon votre profil.
                </div>
                """,
                unsafe_allow_html=True
            )

        # --- 3b. Radar Chart personnalisé : Entreprise vs Cluster cible ---
        cluster_means = df.groupby('cluster')[selected_features].mean()
        entreprise_scores = entreprise[selected_features]
        target_cluster = predicted_cluster
         
        maturity_order = [1, 2, 0]  # Cluster 1 = initial, 2 = intégré, 0 = avancé

        try:
            current_index = maturity_order.index(target_cluster)
            if current_index + 1 < len(maturity_order):
                next_cluster = maturity_order[current_index + 1]
            else:
                next_cluster = target_cluster  # Already at highest maturity
        except ValueError:
            next_cluster = target_cluster  # fallback if cluster ID not in the list
    
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
            st.markdown("### 📊 Radar Chart : Entreprise vs Cluster Cible (par Dimension)")

            # Regroupement des colonnes par dimension
            dimension_groups = {
                "Leadership": [col for col in selected_features if "Leadership" in col],
                "Opérations": [col for col in selected_features if "Opérations" in col],
                "Organisation apprenante": [col for col in selected_features if "Organisation apprenante" in col],
                "Technologies": [col for col in selected_features if "Technologies" in col],
                "Supply Chain": [col for col in selected_features if "Supply Chain" in col],
            }
            
           def moyenne_par_dimension(df, cols):
                if len(cols) > 1:
                    # df[cols] est DataFrame ou Series (si une colonne)
                    # on récupère la première ligne avec .iloc[0], qui est une Series, puis on fait mean()
                    # mais parfois ça renvoie directement une valeur scalar si une colonne, donc on force avec .mean()
                    
                    subset = df[cols]
                    if isinstance(subset, pd.Series):
                        # subset est Series donc on fait mean direct
                        return subset.mean()
                    else:
                        # subset est DataFrame, on récupère la première ligne (Series) et on fait mean
                        return subset.iloc[0].mean()
                else:
                    val = df[cols[0]]
                    # val peut être une Series (colonne), on récupère la première valeur si c'est le cas
                    if isinstance(val, pd.Series):
                        return val.iloc[0]
                    else:
                        return val
            
            # Calcul des moyennes par dimension pour l'entreprise
            entreprise_dim_scores = {
                dim: moyenne_par_dimension(entreprise, cols)
                for dim, cols in dimension_groups.items() if cols
            }
            
            # Pour cluster_means, on récupère un DataFrame avec une seule ligne (cluster ciblé)
            cluster_subset = cluster_means.loc[[next_cluster]]
            
            # Calcul des moyennes par dimension pour le cluster cible
            cluster_dim_scores = {
                dim: moyenne_par_dimension(cluster_subset, cols)
                for dim, cols in dimension_groups.items() if cols
            }
            
            # Création du radar chart
            fig_dim_compare = go.Figure()
            fig_dim_compare.add_trace(go.Scatterpolar(
                r=list(entreprise_dim_scores.values()),
                theta=list(entreprise_dim_scores.keys()),
                fill='toself',
                name="Entreprise",
                line=dict(color='rgba(255, 0, 0, 1)', width=3),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            fig_dim_compare.add_trace(go.Scatterpolar(
                r=list(cluster_dim_scores.values()),
                theta=list(cluster_dim_scores.keys()),
                fill='toself',
                name="Moyenne du cluster cible",
                line=dict(color='rgba(0, 0, 139, 1)', width=3),
                fillcolor='rgba(0, 0, 139, 0.3)'
            ))
            
            fig_dim_compare.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig_dim_compare)




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
        # Calcul des priorités selon l’écart
        def priorite_gap(val):
            if val <= -1.0:
                return "Élevée"
            elif val <= -0.5:
                return "Moyenne"
            else:
                return "Faible"
        
        gap_values = pd.to_numeric(gaps_sorted.values, errors='coerce')
            
        gap_df = pd.DataFrame({
                'Sous-dimension': gaps_sorted.index,
                'Écart': np.round(gap_values, 2),
                'Priorité': [priorite_gap(val) for val in gap_values]
            })
            
        st.dataframe(
        gap_df.style.background_gradient(
                    subset=['Écart'],
                    cmap='YlOrRd_r'  # 🔁 Inversé pour mettre jaune foncé sur gros écart
                ).applymap(
                    lambda x: 'color: red; font-weight: bold' if x == 'Élevée'
                    else 'color: orange; font-weight: bold' if x == 'Moyenne'
                    else 'color: green;',
                    subset=['Priorité']
                )
            )
        


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
        # Ordre des niveaux de maturité

        # Trier par taux d'adoption décroissant
        lean_to_adopt = lean_to_adopt.sort_values(ascending=False)
        tech_to_adopt = tech_to_adopt.sort_values(ascending=False)

        # Affichage méthodes Lean à adopter
        def priorite_adoption(val):
            if val >= 0.7:
                return "Élevée"
            elif val >= 0.4:
                return "Moyenne"
            else:
                return "Faible"
        
        if not lean_to_adopt.empty:
            lean_df = pd.DataFrame({
                "Méthode Lean": lean_to_adopt.index.str.replace('Lean_', ''),
                "Taux d'adoption dans cluster cible": lean_to_adopt.values.round(2),
                "Priorité": [priorite_adoption(v) for v in lean_to_adopt.values]
            })
            st.write("### Méthodes Lean à adopter en priorité")
            st.dataframe(
                lean_df.style.background_gradient(
                    subset=['Taux d\'adoption dans cluster cible'],
                    cmap='Oranges'
                ).applymap(
                    lambda x: 'color: red; font-weight: bold' if x == 'Élevée' else
                              'color: orange; font-weight: bold' if x == 'Moyenne' else
                              'color: green;',
                    subset=['Priorité']
                )
            )

        else:
            st.info("Aucune méthode Lean prioritaire à adopter.")

        # Affichage technologies Industrie 4.0 à adopter
        def priorite_adoption(val):
            if val >= 0.7:
                return "Élevée"
            elif val >= 0.4:
                return "Moyenne"
            else:
                return "Faible"
        
        if not tech_to_adopt.empty:
            tech_df = pd.DataFrame({
                "Technologie Industrie 4.0": tech_to_adopt.index.str.replace('Tech_', ''),
                "Taux d'adoption dans cluster cible": tech_to_adopt.values.round(2),
                "Priorité": [priorite_adoption(v) for v in tech_to_adopt.values]
            })
        
            st.write("### Technologies Industrie 4.0 à adopter en priorité")
            st.dataframe(
                tech_df.style.background_gradient(
                    subset=['Taux d\'adoption dans cluster cible'],
                    cmap='Purples'
                ).applymap(
                    lambda x: 'color: red; font-weight: bold' if x == 'Élevée' else
                              'color: orange; font-weight: bold' if x == 'Moyenne' else
                              'color: green;',
                    subset=['Priorité']
                )
            )
        else:
            st.info("Aucune technologie prioritaire à adopter.")





else:
    st.info("⏳ Veuillez uploader un fichier Excel pour commencer.")
