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
from fpdf import FPDF
import io

st.set_page_config(page_title="Lean 4.0 Cluster & Tree App", layout="wide")
st.title("🔍 Lean 4.0 Clustering & Decision Tree Dashboard")
st.markdown("This app lets you perform clustering and classification on Lean 4.0 survey data using KMeans and Decision Tree models.")

# --- Define dimensions and sub-dimensions ---
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

cols_to_exclude = [
    'Indicateurs suivis', 'Zone investissement principale',
    'Typologie de production', 'Type de flux', 'Pays ',
    'Méthodes Lean ', 'Technologies industrie 4.0', 'cluster',
    'Cluster', 'Feature_Cluster', 'Niveau Maturité', 'Cluster Label'
]

# --- Functions to save figures and create PDF ---

def save_figures_to_files(elbow_fig, silhouette_fig, pca_fig, heatmap_fig, radar_fig):
    elbow_fig.savefig("elbow.png", bbox_inches='tight')
    silhouette_fig.savefig("silhouette.png", bbox_inches='tight')
    pca_fig.savefig("pca.png", bbox_inches='tight')
    heatmap_fig.savefig("heatmap.png", bbox_inches='tight')
    radar_fig.write_image("radar.png")

def create_pdf_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rapport Lean 4.0 - Clustering & Analyse", ln=True, align='C')

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Méthode Elbow", ln=True)
    pdf.image("elbow.png", w=180)
    
    pdf.cell(0, 10, "Score de Silhouette", ln=True)
    pdf.image("silhouette.png", w=180)

    pdf.cell(0, 10, "Visualisation PCA", ln=True)
    pdf.image("pca.png", w=180)

    pdf.cell(0, 10, "Profil des Clusters (Radar Chart)", ln=True)
    pdf.image("radar.png", w=180)

    pdf.cell(0, 10, "Carte Thermique des Scores Moyens", ln=True)
    pdf.image("heatmap.png", w=180)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# --- Sidebar: Upload file ---
st.sidebar.header("📂 Upload your CSV file")
file = st.sidebar.file_uploader("Upload df_cleaned_with_dummies.csv", type="csv")

if file:
    df = pd.read_csv(file)
    st.success("✅ File loaded successfully")

    # --- Sidebar: Select sub-dimensions for clustering ---
    st.sidebar.markdown("### 📌 Sélectionner les sous-dimensions par dimension")
    selected_features = []
    for dimension, sub_dims in dimension_map.items():
        with st.sidebar.expander(f"🧩 {dimension}"):
            selected = st.multiselect(f"Sous-dimensions de {dimension}", sub_dims, default=sub_dims, key=dimension)
            selected_features.extend(selected)

    if not selected_features:
        st.warning("⚠️ Veuillez sélectionner au moins une sous-dimension pour continuer.")
        st.stop()

    # --- Scale features ---
    features = df[selected_features].dropna()
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

    # Elbow plot
    fig1, ax1 = plt.subplots()
    ax1.plot(range(k_range[0], k_range[1] + 1), inertia, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    st.pyplot(fig1)
    elbow_fig = fig1

    # Silhouette plot
    fig2, ax2 = plt.subplots()
    ax2.plot(range(k_range[0], k_range[1] + 1), silhouette_scores, marker='o')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Score')
    st.pyplot(fig2)
    silhouette_fig = fig2

    final_k = st.selectbox("Select final K", list(range(k_range[0], k_range[1] + 1)))
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # Map cluster number to maturity level label
    niveau_maturite_map = {
        0: 'Niveau Avancé',
        1: 'Niveau Initial',
        2: 'Niveau Intégré'
    }
    df['Niveau de maturité Lean 4.0'] = df['cluster'].map(niveau_maturite_map)

    st.subheader("📋 Cluster Analysis")
    cluster_counts = df['cluster'].value_counts().sort_index()
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
    df_pca['Niveau de maturité Lean 4.0'] = df['Niveau de maturité Lean 4.0']

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Niveau de maturité Lean 4.0', palette='viridis', ax=ax3)
    ax3.set_title("PCA of Clusters")
    st.pyplot(fig3)
    pca_fig = fig3

    # Heatmap
    avg_scores = df.groupby('Niveau de maturité Lean 4.0')[selected_features].mean()
    st.subheader("📈 Average Survey Scores per Cluster (Heatmap)")
    fig, ax = plt.subplots(figsize=(14, max(9, len(selected_features)*0.5)))
    sns.heatmap(
        avg_scores.T,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor='gray',
        cbar_kws={'label': 'Average Score', 'shrink': 0.75, 'aspect': 15, 'pad': 0.02},
        ax=ax
    )
    ax.set_title("Average Survey Scores per Cluster", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Niveau de maturité Lean 4.0", fontsize=14, labelpad=15)
    ax.set_ylabel("Survey Features", fontsize=14, labelpad=15)
    ax.set_yticks(np.arange(len(selected_features)) + 0.5)
    ax.set_yticklabels(selected_features, fontsize=12, rotation=0)
    ax.tick_params(axis='x', labelsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    heatmap_fig = fig

    # --- Sidebar: select dimensions to display in radar ---
    st.sidebar.markdown("### 🎯 Choisissez les dimensions à afficher dans le Radar Chart")
    all_dimensions = list(dimension_map.keys())
    selected_dimensions_for_radar = st.sidebar.multiselect(
        "Dimensions pour Radar Chart",
        options=all_dimensions,
        default=all_dimensions,
        key='radar_dimensions'
    )

    selected_features_for_radar = []
    for dim in selected_dimensions_for_radar:
        selected_features_for_radar.extend(dimension_map[dim])
    selected_features_for_radar = [f for f in selected_features_for_radar if f in selected_features]  # only keep those selected for clustering

    # --- Radar Chart ---
    tabs = st.tabs(["Clustering", "Decision Tree", "Radar Chart"])
    with tabs[2]:
        st.header("📡 Radar Chart - Profils par Dimension")
        try:
            cluster_avg = df.groupby('Niveau de maturité Lean 4.0')[selected_features_for_radar].mean().dropna(axis=1, how='any')
            available_features = cluster_avg.columns.tolist()

            custom_colors = {
                'Niveau Initial': {
                    'line': 'rgba(0, 0, 139, 1)',       # Dark Blue opaque
                    'fill': 'rgba(0, 0, 139, 0.5)'      # Dark Blue semi-transparent
                },
                'Niveau Avancé': {
                    'line': 'rgba(173, 216, 230, 1)',   # Light Blue
                    'fill': 'rgba(173, 216, 230, 0.3)'  # Transparent Light Blue
                },
                'Niveau Intégré': {
                    'line': 'rgba(255, 0, 0, 1)',       # Red
                    'fill': 'rgba(255, 0, 0, 0.3)'      # Transparent Red
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

    # --- Decision Tree tab ---
    with tabs[1]:
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
            y_pred = clf.predict(X_test)

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
            from sklearn.tree import export_graphviz
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

    # --- Export PDF report button ---
    st.markdown("---")
    if st.button("📥 Exporter le rapport PDF"):
        try:
            save_figures_to_files(elbow_fig, silhouette_fig, pca_fig, heatmap_fig, fig_radar)
            pdf_data = create_pdf_report()
            st.download_button("Télécharger le rapport PDF", pdf_data, file_name="rapport_lean40.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erreur lors de la génération du rapport PDF : {e}")

else:
    st.info("👈 Upload a file to begin.")
