# 🔁 Assuming all previous sections including model loading and definitions exist
# This part is added in your "Application" section (e.g., inside `with tabs[5]:`)

# Example company (Entreprise 5) features
entreprise = pd.Series({
    'Leadership - Engagement Lean': 4.0,
    'Leadership - Engagement DT': 2.0,
    'Leadership - Stratégie': 2.0,
    'Leadership - Communication': 3.0,
    'Supply Chain - Collaboration inter-organisationnelle': 3.0,
    'Supply Chain - Traçabilité': 2.0,
    'Supply Chain - Impact sur les employées': 3.0,
    'Opérations - Standardisation des processus': 2.0,
    'Opérations - Juste-à-temps (JAT)': 3.0,
    'Opérations - Gestion des résistances': 2.0,
    'Technologies - Connectivité et gestion des données': 3.0,
    'Technologies - Automatisation': 2.0,
    'Technologies - Pilotage du changement': 3.0,
    'Organisation apprenante - Formation et développement des compétences': 3.0,
    'Organisation apprenante - Collaboration et Partage des Connaissances': 3.0,
    'Organisation apprenante - Flexibilité organisationnelle': 3.0
})

# Predict cluster using KMeans
cluster_pred = kmeans_model.predict([entreprise.values])[0]

# Create dummy DataFrame for technologies/methods used
used_features = {
    'QRQC': 1,
    '5S': 1,
    'Value Stream Mapping (VSM)': 1,
    'TPM / TRS method': 1,
    'Takt Time': 1,
    'Intelligence Artificielle': 1,
    'ERP (Enterprise Resource Planning)': 1,
    'Kaizen': 0,
    '6 sigma': 0,
    'Kanban': 0,
    'Poka Yoke': 0
    # ... Add other features used in Decision Tree here
}
features_dt_new = pd.DataFrame([used_features])

# Predict maturity level using Decision Tree
predicted_level = model_dt.predict(features_dt_new.select_dtypes(include=[np.number]).fillna(0))[0]

# Get real maturity level name
cluster_name = {0: "Niveau Initial", 1: "Niveau Intégré", 2: "Niveau Avancé"}
real_level = cluster_name[cluster_pred]

st.subheader("🧠 Scénario de l'entreprise")
st.write(f"Niveau réel (par clustering): **{real_level}**")
st.write(f"Niveau prédit (par arbre de décision): **{predicted_level}**")

# Calculate gaps compared to the target cluster (e.g. cluster 2 / Niveau Intégré)
cluster_avg = df.groupby('Maturité Label')[entreprise.index].mean()
top_gaps = cluster_avg.loc['Niveau Intégré'] - entreprise

# Keep top 5 largest negative gaps
top_gaps = top_gaps.sort_values().head(5)

# Extract Dimension and Sous-dimension from the index
dims = []
subs = []
for name in top_gaps.index:
    parts = name.split(" - ")
    if len(parts) == 2:
        dims.append(parts[0])
        subs.append(parts[1])
    else:
        dims.append(name)
        subs.append("")

# Display roadmap for sub-dimension improvement
gap_df = pd.DataFrame({
    'Dimension': dims,
    'Sous-dimension': subs,
    'Score Entreprise': entreprise.loc[top_gaps.index].values,
    'Moyenne Cluster 2': cluster_avg.loc['Niveau Intégré', top_gaps.index].values,
    'Écart': top_gaps.round(2).values
})

st.subheader("📉 Feuille de route - Sous-dimensions prioritaires")
st.dataframe(gap_df)

# Technologies/méthodes à recommander (exemple de DataFrame prêt à afficher)
to_adopt = pd.Series({
    'Kaizen': 0.71,
    '6 sigma': 0.71,
    'Kanban': 0.54,
    'TPM / TRS méthode': 0.46,
    'Poka Yoke': 0.38,
    'MES': 0.46,
    'Big Data et Analytics': 0.25,
    'Robots autonomes': 0.29,
    'Simulation': 0.25
    # ... add more if needed
})

techno_lean_df = pd.DataFrame({
    'Outil / Technologie': to_adopt.index,
    'Taux d\'adoption Cluster 2': to_adopt.values,
    'Utilisation par l\'entreprise': ['Non'] * len(to_adopt)
})

st.subheader("🔧 Feuille de route technologique personnalisée")
st.dataframe(techno_lean_df)

# Done 🎉
