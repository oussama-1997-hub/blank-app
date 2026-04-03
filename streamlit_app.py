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
    <style>
    hide_github_icon = “”"
        
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
        “”"
        st.markdown(hide_github_icon, unsafe_allow_html=True)
    </style>
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


use_github = st.sidebar.checkbox("📂 Use database from GitHub instead of uploading")

if use_github:
    # GitHub raw URL for Excel file
    github_url = "https://raw.githubusercontent.com/oussama-1997-hub/blank-app/main/df_cleaned_with_dummies.xlsx"

    @st.cache_data
    def load_data():
        return pd.read_excel(github_url)
    
    df = load_data()
    st.success("✅ Excel file loaded successfully from GitHub!")

    
    from PIL import Image
    import base64
    from io import BytesIO
    
    # Charger et redimensionner l'image
    image = Image.open("MM lean 4.0.png")
    resized_image = image.resize((int(image.width * 0.8), int(image.height * 0.8)), Image.LANCZOS)
    
    # Convertir en base64 pour l'affichage HTML
    buffered = BytesIO()
    resized_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Affichage centré
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{img_b64}" style="width:80%; max-width:900px;">
            <p style='font-size:18px; color:#444;'>🧭 Modèle de Maturité Lean 4.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # df['Lean_Méthode TPM / TRS'] = df['Lean_Méthode TPM / TRS'] | df['Lean_TPM / TRS method']
    # df.drop(columns=['Lean_TPM / TRS method'], inplace=True)

    # df['Lean_DDMRP/ hoshin kanri'] = (
    #     df['Lean_DDMRP/ hoshin kanri'] |
    #     df['Lean_DDMRP'] |
    #     df['Lean_Maki-Gami/Hoshin…etc']
    # )
    # df.drop(columns=['Lean_DDMRP', 'Lean_Maki-Gami/Hoshin…etc'], inplace=True)
    
    # # Tech tools
    # df['Tech_Réalité augmentée'] = df['Tech_Réalité augmentée'] | df['Tech_Augmented reality']
    # df.drop(columns=['Tech_Augmented reality'], inplace=True)
    
    # df['Tech_Systèmes cyber physiques'] = df['Tech_Systèmes cyber physiques'] | df['Tech_Cyber ​​physical systems']
    # df.drop(columns=['Tech_Cyber ​​physical systems'], inplace=True)
    
    # df['Tech_Intelligence artificielle'] = df['Tech_Intelligence artificielle'] | df['Tech_Artificial intelligence']
    # df.drop(columns=['Tech_Artificial intelligence'], inplace=True)
    
    # df['Tech_Robots autonomes'] = df['Tech_Robots autonomes'] | df['Tech_Autonomous robots']
    # df.drop(columns=['Tech_Autonomous robots'], inplace=True)

    # column_to_drop = 'Tech_Je ne sais pas'
    # df.drop(columns=[column_to_drop], inplace=True)
    
    
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
    available_selected_features = [col for col in selected_features if col in df.columns]
    if len(available_selected_features) < len(selected_features):
        missing = list(set(selected_features) - set(df.columns))
        st.warning(f"⚠️ Colonnes manquantes dans le fichier : {missing}")
    if not available_selected_features:
        st.stop()
    
    features = df[available_selected_features].dropna()
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

        k_values = list(range(k_range[0], k_range[1] + 1))
        
        # Ensure 3 is inside the range
        default_k = 3 if 3 in k_values else k_values[0]
        
        final_k = st.selectbox(
            "Select final K",
            k_values,
            index=k_values.index(default_k)
        )
        
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
        st.header("📡 Radar Chart - Profils par *Sous-Dimension*")
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
        avg_scores = df.groupby('Cluster')[selected_features].mean()

        # Detect Lean and Tech dummy columns
        tech_cols = [col for col in df.columns if col.startswith('Tech_')]
        lean_cols = [col for col in df.columns if col.startswith('Lean_')]

        lean_avg = df.groupby('Cluster')[lean_cols].mean() if lean_cols else pd.DataFrame()
        tech_avg = df.groupby('Cluster')[tech_cols].mean() if tech_cols else pd.DataFrame()

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
        # Lean methods
        df['Lean_Méthode TPM / TRS'] = df['Lean_Méthode TPM / TRS'] | df['Lean_TPM / TRS method']
        df.drop(columns=['Lean_TPM / TRS method'], inplace=True)
        
        df['Lean_DDMRP/ hoshin kanri'] = (
            df['Lean_DDMRP/ hoshin kanri'] |
            df['Lean_DDMRP'] |
            df['Lean_Maki-Gami/Hoshin…etc']
        )
        df.drop(columns=['Lean_DDMRP', 'Lean_Maki-Gami/Hoshin…etc'], inplace=True)
        
        df['Lean_Juste à temps'] = df['Opérations - Juste-à-temps (JAT)'].apply(lambda x: 1 if x in [4, 5] else 0)
        df['Lean_Just in time'] = df['Lean_Juste à temps'] | df['Lean_Just in time']
        df.drop(columns=['Lean_Just in time'], inplace=True)
        
        # Tech tools
        df['Tech_Réalité augmentée'] = df['Tech_Réalité augmentée'] | df['Tech_Augmented reality']
        df.drop(columns=['Tech_Augmented reality'], inplace=True)
        
        df['Tech_Systèmes cyber physiques'] = df['Tech_Systèmes cyber physiques'] | df['Tech_Cyber ​​physical systems']
        df.drop(columns=['Tech_Cyber ​​physical systems'], inplace=True)
        
        df['Tech_Intelligence artificielle'] = df['Tech_Intelligence artificielle'] | df['Tech_Artificial intelligence']
        df.drop(columns=['Tech_Artificial intelligence'], inplace=True)
        
        df['Tech_Robots autonomes'] = df['Tech_Robots autonomes'] | df['Tech_Autonomous robots']
        df.drop(columns=['Tech_Autonomous robots'], inplace=True)
        #target_col = 'Niveau de maturité Lean 4.0'
        target_col = 'Niveau Maturité'
        target = 'Niveau Maturité'
        if target_col in df.columns:
            # Columns to remove based on prefix
            # cols_prefix_to_remove = df.filter(regex=r'^(Secteur|taille)').columns.tolist()
            # exclude_cols = (
            # exclude_cols + cols_prefix_to_remove
            # )
            features_dt = df.drop(columns=exclude_cols, errors='ignore')
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
                "Organisation apprenante  - Flexibilité organisationnelle",
            ]
            # Define the columns to remove from features
            cols_to_remove_from_features = [
                'Indicateurs suivis',
                'Zone investissement principale',
                'Typologie de production',
                'Type de flux',
                'Pays ',
                'Cluster Label',
                'taille_categorie', 'Secteur_Groupe', 'taille_categorie_encoded',
                'Secteur_Groupe_Divers', 'Secteur_Groupe_Industrie lourde',
                'Secteur_Groupe_Industrie légère',
                'Secteur_Groupe_Service et Technologie'
            ] + colonnes # Add the original survey question columns
            # Columns to remove based on prefix
            cols_prefix_to_remove = [
                col for col in df.columns
                if col.startswith('Secteur') or col.startswith('taille')
            ]
            cols_to_remove_from_features = (
                cols_to_remove_from_features + cols_prefix_to_remove
            )
            # Create the feature set by dropping the target and specified columns from df
            # Make sure to drop the original multi-value columns ('Méthodes Lean ', 'Technologies industrie 4.0') as they are replaced by cledf_cleaed_clustered
            features = df.drop(columns=[target] + cols_to_remove_from_features + ['Méthodes Lean ', 'Technologies industrie 4.0', 'cluster', 'Cluster', 'Feature_Cluster', 'Niveau Maturité','Taille entreprise ','Secteur industriel',"Niveau de maturité Lean 4.0"], errors='ignore')
            
            # Ensure there are no remaining non-numeric columns that weren't intended to be features
            # For robustness, drop any remaining object type columns if they exist unexpectedly
            features = features.select_dtypes(exclude=['object'])
            
            # Also, ensure there are no NaNs in the features or target
            features.fillna(0, inplace=True) # Fill potential NaNs with 0 (common for dummy variables)
            y = df[target].dropna()
            features = features.loc[y.index] # Align features with the non-null target values
            
            # Separate features (X) and target (y)
            X = features
            y = df[target_col]
            features_dt = features_dt.loc[y.index]



            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46, stratify=y)
            clf = DecisionTreeClassifier(random_state=10, min_samples_leaf=3, max_depth=5)
            clf.fit(X_train, y_train)
                        # Display columns in Streamlit

            
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

        # === Affichage des scores de maturité (par groupe de sous-dimensions) ===
                
        # Secteur et taille de l'entreprise sélectionnée
        secteur = entreprise["Secteur industriel"] if "Secteur industriel" in entreprise.index else "N/A"
        taille  = entreprise["Taille entreprise "] if "Taille entreprise " in entreprise.index else "N/A"
        
                
                
        st.markdown("### 🏭 Informations générales sur l'entreprise")
        st.markdown(f"- **Secteur d'activité :** {secteur}")
        st.markdown(f"- **Taille de l'entreprise :** {taille}")
        st.write("---")

        entreprise_features = entreprise[selected_features].values.flatten()
        scores_dict = dict(zip(selected_features, entreprise_features))
        st.markdown("### 📊 Scores de maturité par sous-dimension")
        # Groupes par dimension (selon les noms réels)
        groupes = {
            "Stratégie - Leadership": [
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
        
        # Construction du tableau final
        rows = []
        for dim, sous_dims in groupes.items():
            for sd in sous_dims:
                score = scores_dict.get(sd, "N/A")
                rows.append({
                    "Dimension": dim,
                    "Sous-dimension": sd.strip(),
                    "Score": round(score, 2) if isinstance(score, (int, float)) else score
                })
        
        df_scores = pd.DataFrame(rows)
        
        # Affichage stylisé
        def stylize(df):
            return df.style.set_properties(**{
                'text-align': 'center',
                'vertical-align': 'middle',
                'font-size': '14px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f0f0f0')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}
            ])
        
        st.dataframe(stylize(df_scores), use_container_width=True)


        st.markdown("### 🛠️ Méthodes Lean & Technologies Industrie 4.0 adoptées")
        
        # Identification des colonnes binaires
        lean_cols = [col for col in df.columns if col.startswith('Lean_')]
        tech_cols = [col for col in df.columns if col.startswith('Tech_')]
        
        # Liste des méthodes/tech adoptées
        lean_adopted = [col.replace('Lean_', '') for col in lean_cols if entreprise.get(col, 0) == 1]
        tech_adopted = [col.replace('Tech_', '') for col in tech_cols if entreprise.get(col, 0) == 1]
        
        # Création des tableaux stylisés
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Méthodes Lean utilisées")
            if lean_adopted:
                lean_df = pd.DataFrame({"Méthode Lean": lean_adopted})
                st.dataframe(lean_df.style
                             .set_properties(**{'text-align': 'center'})
                             .set_table_styles([{
                                 'selector': 'th',
                                 'props': [('text-align', 'center'), ('background-color', '#f5f5f5')]
                             }]),
                             use_container_width=True)
            else:
                st.info("Aucune méthode Lean détectée.")
        
        with col2:
            st.markdown("#### ✅ Technologies Industrie 4.0 utilisées")
            if tech_adopted:
                tech_df = pd.DataFrame({"Technologie 4.0": tech_adopted})
                st.dataframe(tech_df.style
                             .set_properties(**{'text-align': 'center'})
                             .set_table_styles([{
                                 'selector': 'th',
                                 'props': [('text-align', 'center'), ('background-color', '#f5f5f5')]
                             }]),
                             use_container_width=True)
            else:
                st.info("Aucune technologie 4.0 détectée.")


            
        # --- 1. Prédiction cluster KMeans (niveau réel) ---
        entreprise_scaled = scaler.transform(entreprise[selected_features].values.reshape(1, -1))
        predicted_cluster = kmeans.predict(entreprise_scaled)[0]
        predicted_cluster_label = cluster_label_map.get(predicted_cluster, "Inconnu")
       
        st.write(f"**Niveau de maturité organisationnelle prédite via K-means sur la base du modèle de maturité :** {predicted_cluster_label}")

        # --- 2. Prédiction arbre de décision (niveau prédit) ---
        # Préparer features DT (technos et lean dummies)
        # Suppose entreprise is a Series (like df.loc[5])
        features_dt_new = pd.DataFrame([entreprise]).drop(columns=exclude_cols, errors='ignore')
        features_dt_new.columns = [col.strip() for col in features_dt_new.columns]  # optional cleaning
        features_dt_new = features_dt_new.select_dtypes(include=[np.number]).fillna(0)
        features_dt_array = features_dt_new.values.reshape(1, -1)  # final input to model
        
        predicted_dt = clf.predict(features_dt_new)[0]
        st.write(f"**Niveau de maturité technologique prédite par arbre de décision selon technologies avancées et méthodes Lean adoptées:** {predicted_dt}")

        # --- 3. Analyse comparative & scénarios ---
        # --- Analyse comparative et recommandations ---
        with st.container():
            st.markdown("## 🧭 Guide d’utilisation personnalisé", unsafe_allow_html=True)
            st.write("---")
        
            st.markdown("""
            <div style="background-color:#f9f9f9;padding:15px;border-radius:10px;">
            <h4>🧩 Ordre de priorité à suivre dans votre démarche Lean 4.0</h4>
            <ul>
                <li><b>Étape 1 :</b> Identification de votre scénario d’adoption (retard technologique / alignement / retard organisationnel)</li>
                <li><b>Étape 2 :</b> Repérage des premières actions prioritaires à mener via les <b>nœuds parents de l’arbre de décision</b></li>
                <li><b>Étape 3 :</b> Génération de <b>feuilles de route personnalisées</b> :
                    <ul>
                        <li>📈 <b>Technologique</b> : Technologies & méthodes Lean à adopter en priorité</li>
                        <li>🧱 <b>Maturité Lean 4.0</b> : Sous-dimensions organisationnelles à améliorer en priorité</li>
                    </ul>
                </li>
                <li><b>Étape 4 :</b> Suivi et mise en œuvre progressive de ces feuilles de route en fonction du scénario identifié</li>
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
         
        # Ordre de maturité réel
        cluster_maturity_rank = {
            1: 1,  # Initial
            2: 2,  # Intégré
            0: 3   # Avancé
        }
        
        available_clusters = cluster_means.index.tolist()
        
        current_rank = cluster_maturity_rank.get(target_cluster, None)
        
        # Clusters plus matures et existants
        higher_clusters = [
            c for c in available_clusters
            if cluster_maturity_rank.get(c, 0) > current_rank
        ]
        
        # Sélection du cluster cible
        if higher_clusters:
            next_cluster = min(
                higher_clusters,
                key=lambda c: cluster_maturity_rank[c]
            )
        else:
            next_cluster = target_cluster  # déjà au max possible

    
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
            
            dimension_groups = {
                "Leadership": [col for col in selected_features if "Leadership" in col],
                "Opérations": [col for col in selected_features if "Opérations" in col],
                "Organisation apprenante": [col for col in selected_features if "Organisation apprenante" in col],
                "Technologies": [col for col in selected_features if "Technologies" in col],
                "Supply Chain": [col for col in selected_features if "Supply Chain" in col],
            }
            
            def moyenne_par_dimension(df, cols):
                if len(cols) > 1:
                    subset = df[cols]
                    if isinstance(subset, pd.Series):
                        return subset.mean()
                    else:
                        return subset.iloc[0].mean()
                else:
                    val = df[cols[0]]
                    if isinstance(val, pd.Series):
                        return val.iloc[0]
                    else:
                        return val
            
            entreprise_dim_scores = {
                dim: moyenne_par_dimension(entreprise, cols)
                for dim, cols in dimension_groups.items() if cols
            }
            
            cluster_subset = cluster_means.loc[[next_cluster]]
            
            cluster_dim_scores = {
                dim: moyenne_par_dimension(cluster_subset, cols)
                for dim, cols in dimension_groups.items() if cols
            }
            
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
                    gap_df.style
                        .background_gradient(subset=['Écart'], cmap='YlOrRd_r')
                        .map(
                            lambda x: (
                                'color: red; font-weight: bold' if str(x).strip().lower() == 'élevée'
                                else 'color: orange; font-weight: bold' if str(x).strip().lower() == 'moyenne'
                                else 'color: green;'
                            ),
                            subset=['Priorité']
                        )
                )

        # 4b. Feuille de route technologique personnalisée
        st.subheader("Méthodes Lean & Technologies à adopter")
        # Dictionnaire : Méthode Lean → Technologies support et interprétation
        lean_support = {
            "Juste à temps (JAT)": {
                "Technologies": "Robots autonomes, WMS, RFID",
                "Interprétation": "Les robots et WMS automatisent la logistique interne, tandis que la RFID assure un suivi en temps réel des flux."
            },
            "Takt Time": {
                "Technologies": "Big Data & Analytics, Systèmes cyber-physiques, ERP, WMS",
                "Interprétation": "L’analyse des données permet d’ajuster le Takt Time selon la demande. Les autres technologies permettent la synchronisation."
            },
            "Heijunka": {
                "Technologies": "WMS, MES",
                "Interprétation": "Le lissage de la production repose sur une gestion fine des stocks (WMS) et le suivi des exécutions (MES)."
            },
            "TPM / TRS": {
                "Technologies": "MES, RFID",
                "Interprétation": "MES et RFID permettent de surveiller la disponibilité des équipements, facilitant la mise en œuvre du TPM."
            },
            "Poka Yoke": {
                "Technologies": "Simulation, Robots autonomes, ERP",
                "Interprétation": "Simulation pour concevoir sans erreurs, robots pour tâches répétitives, ERP pour intégrer les contrôles qualité."
            },
            "Kaizen": {
                "Technologies": "MES, RFID, Big Data & Analytics, Fabrication additive (Impression 3D)",
                "Interprétation": "Ces technologies soutiennent les cycles Kaizen en automatisant les suivis et en accélérant les tests."
            },
            "Kanban": {
                "Technologies": "Fabrication additive (Impression 3D)",
                "Interprétation": "L’impression 3D permet une production réactive pour alimenter un système Kanban flexible."
            },
            "Value Stream Mapping (VSM)": {
                "Technologies": "Systèmes cyber-physiques, RFID, WMS",
                "Interprétation": "Ces technologies enrichissent la VSM avec des données terrain sur les flux physiques et stocks."
            },
            "QRQC": {
                "Technologies": "Intelligence artificielle",
                "Interprétation": "L’IA aide à détecter automatiquement les anomalies, renforçant l’efficacité des boucles QRQC."
            }
        }

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
        # --- Lean mapping for display names ---
        lean_to_tech_support = {
            "Juste à temps": "Robots autonomes, WMS, RFID",
            "Takt Time": "Big Data & Analytics, Systèmes cyber-physiques, ERP, WMS",
            "Heijunka": "WMS, MES",
            "Méthode TPM / TRS": "MES, RFID",
            "Poka Yoke": "Simulation, Robots autonomes, ERP",
            "Kaizen": "MES, RFID, Big Data & Analytics, Fabrication additive (Impression 3D)",
            "Kanban": "Fabrication additive (Impression 3D)",
            "Value Stream Mapping (VSM)": "Systèmes cyber-physiques, RFID, WMS",
            "QRQC": "Intelligence artificielle"
        }

        mapping_lean_columns_to_display = {
            'Lean_QRQC': 'QRQC',
            'Lean_DDMRP/ hoshin kanri': 'DDMRP / Hoshin Kanri',
            'Lean_5S': '5S',
            'Lean_Heijunka': 'Heijunka',
            'Lean_Maki-Gami/Hoshin…etc': 'Maki-Gami / Hoshin',
            'Lean_Value Stream Mapping (VSM)': 'Value Stream Mapping (VSM)',
            'Lean_Kaizen': 'Kaizen',
            'Lean_DDMRP': 'DDMRP',
            'Lean_Méthode TPM / TRS': 'Méthode TPM / TRS',
            'Lean_Kata': 'Kata',
            'Lean_Just in time': 'Juste à temps (JAT)',
            'Lean_QRAP': 'QRAP',
            'Lean_TPM / TRS method': 'TPM / TRS',
            'Lean_6 sigma': '6 Sigma',
            'Lean_Poka Yoke': 'Poka Yoke',
            'Lean_Takt Time': 'Takt Time',
            'Lean_Kanban': 'Kanban',
            'Lean_GEMBA': 'Gemba'
        }
       
        # Create display names list for lean_to_adopt
        lean_methods_display = [mapping_lean_columns_to_display.get(col, col.replace('Lean_', '')) for col in lean_to_adopt.index]
        
        # Create support tech list matching display names or empty if not found
        technologies_support = [lean_to_tech_support.get(method, "") for method in lean_methods_display]
        
     # Définition de la fonction priorite_adoption
        def priorite_adoption(val):
            if val >= 0.7:
                return "Élevée"
            elif val >= 0.4:
                return "Moyenne"
            else:
                return "Faible"
        
        # Supposons que lean_to_adopt, lean_methods_display, et technologies_support sont déjà calculés correctement
        # lean_methods_display = lean_to_adopt.index.str.replace('Lean_', '').tolist()
        # technologies_support = [lean_to_tech_support.get(meth, "") for meth in lean_methods_display]
        
        # Construire le DataFrame
        lean_df = pd.DataFrame({
            "Méthode Lean": lean_methods_display,
            "Technologies support": technologies_support,
            "Taux d'adoption dans cluster cible": lean_to_adopt.values.round(2),
            "Priorité": [priorite_adoption(v) for v in lean_to_adopt.values]
        })
        
        # Display the styled DataFrame
        st.markdown("### 🛠️ Méthodes Lean à adopter en priorité")
        styled_lean_df = lean_df.style\
            .background_gradient(
                subset=["Taux d'adoption dans cluster cible"], 
                cmap="Oranges"
            )\
            .map(
                lambda x: 'color: red; font-weight: bold' if x == 'Élevée' else
                          'color: orange; font-weight: bold' if x == 'Moyenne' else
                          'color: green;',
                subset=["Priorité"]
            )
            .set_properties(**{'text-align': 'center'})\
            .set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center')]
            }])
        st.dataframe(styled_lean_df, use_container_width=True)
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
