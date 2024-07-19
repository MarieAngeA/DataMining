import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from io import StringIO

# Fonctions utilitaires pour la suppression des lignes/colonnes
def remove_duplicates(data):
    return data.drop_duplicates()

def remove_empty_rows(data):
    return data.dropna(how='all')

def remove_high_missing_rows(data, threshold=0.8):
    return data.dropna(thresh=int(threshold * data.shape[1]))

def remove_high_missing_cols(data, threshold=0.8):
    return data.dropna(axis=1, thresh=int(threshold * data.shape[0]))

def handle_missing_values(data, method):
    if method == "Remplacer par la moyenne":
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[num_cols] = data[num_cols].apply(lambda col: col.fillna(col.mean()))
    elif method == "Remplacer par la médiane":
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[num_cols] = data[num_cols].apply(lambda col: col.fillna(col.median()))
    elif method == "Remplacer par le mode":
        data = data.apply(lambda col: col.fillna(col.mode()[0]))
    return data

def normalize_data(data, method):
    if method == "Min-Max":
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])
    elif method == "Z-Score":
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

def visualize_data(data, column):
    st.subheader("Histogramme")
    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(x=data[column], ax=ax)
    st.pyplot(fig)

def perform_pca(data):
    st.header("Analyse en Composantes Principales (PCA)")
    try:
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(data.select_dtypes(include=['float64', 'int64']).dropna())
        pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
        st.write(pca_df)

        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'])
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors du calcul du PCA : {e}")

def perform_lda(data):
    st.header("Analyse Discriminante Linéaire (LDA)")
    if 'Cluster' in data.columns:
        try:
            lda = LDA(n_components=2)
            lda_components = lda.fit_transform(data.select_dtypes(include=['float64', 'int64']).dropna(), data['Cluster'])
            lda_df = pd.DataFrame(data=lda_components, columns=['LDA1', 'LDA2'])
            st.write(lda_df)

            fig, ax = plt.subplots()
            scatter = ax.scatter(lda_df['LDA1'], lda_df['LDA2'], c=data['Cluster'], cmap='viridis')
            ax.set_xlabel('LDA1')
            ax.set_ylabel('LDA2')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors du calcul du LDA : {e}")
    else:
        st.write("L'Analyse Discriminante Linéaire (LDA) nécessite des clusters. Veuillez effectuer le clustering d'abord.")

def perform_clustering(data):
    st.header("Clustering des Données")
    clustering_method = st.selectbox(
        "Sélectionnez une méthode de clustering",
        ("K-Means", "DBSCAN")
    )

    if clustering_method == "K-Means":
        num_clusters = st.slider("Nombre de clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters)
        data['Cluster'] = kmeans.fit_predict(data.select_dtypes(include=['float64', 'int64']).dropna())
    elif clustering_method == "DBSCAN":
        eps = st.slider("EPS", 0.1, 10.0, 0.5)
        min_samples = st.slider("Nombre minimum d'échantillons", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        data['Cluster'] = dbscan.fit_predict(data.select_dtypes(include(['float64', 'int64']).dropna()))

    st.write("Aperçu des clusters :")
    st.write(data)

    st.header("Visualisation des Clusters")
    try:
        pca = PCA(n_components=2)
        components = pca.fit_transform(data.select_dtypes(include=['float64', 'int64']).dropna())

        fig, ax = plt.subplots()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=data['Cluster'], cmap='viridis')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        st.pyplot(fig)

        st.write("Statistiques des clusters :")
        st.write(data.groupby('Cluster').size().reset_index(name='Count'))
    except Exception as e:
        st.error(f"Erreur lors de la visualisation des clusters : {e}")

def analyze_data_distribution(data):
    st.header("Analyse de la Distribution des Données")
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            st.write(f"**{column}**")
            st.write(f"Type: Numérique")
            skewness = skew(data[column].dropna())
            kurt = kurtosis(data[column].dropna())
            st.write(f"Asymétrie (Skewness): {skewness}")
            st.write(f"Aplatissement (Kurtosis): {kurt}")
            if skewness > -0.5 and skewness < 0.5:
                st.write("Distribution: Symétrique")
            elif skewness <= -0.5:
                st.write("Distribution: Asymétrique à gauche (Négative)")
            else:
                st.write("Distribution: Asymétrique à droite (Positive)")
            st.write("---")
        else:
            st.write(f"**{column}**")
            st.write(f"Type: Catégorique")
            st.write("Valeurs uniques:")
            st.write(data[column].value_counts())
            st.write("---")

# Titre de l'application
st.title('Application d\'Exploration et de Clustering des Données')

# Téléchargement du fichier
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

# Options pour sélectionner l'entête et le séparateur
header_option = st.selectbox("Ligne d'entête", [None, 0, 1, 2, 3], index=1)
separator_option = st.selectbox("Type de séparateur", [",", ";", "\t", "|"], index=0)

# Liste des encodages courants
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
encoding_option = st.selectbox("Choisissez un encodage", encodings, index=0)

if uploaded_file is not None:
    try:
        # Lire le contenu du fichier en tant que bytes
        bytes_data = uploaded_file.getvalue()

        # Convertir en StringIO
        stringio = StringIO(bytes_data.decode(encoding_option))

        # Lire les données en tant que chaîne de caractères
        string_data = stringio.read()

        # Utiliser pandas pour lire le fichier CSV avec les options spécifiées
        try:
            df = pd.read_csv(StringIO(string_data), header=header_option, delimiter=separator_option)
        except pd.errors.ParserError as e:
            st.error(f"Erreur de parsing du fichier : {e}")
            st.stop()
        except UnicodeDecodeError as e:
            st.error(f"Erreur d'encodage du fichier : {e}")
            st.stop()

        # Stocker le DataFrame initial dans st.session_state
        st.session_state.df = df

        st.write("DataFrame :", df.head())

        # Affichage des premières et dernières lignes du fichier
        st.header("Aperçu des Données")
        st.write("Premières lignes du dataset :")
        st.write(df.head())

        st.write("Dernières lignes du dataset :")
        st.write(df.tail())

        # Résumé statistique de base
        st.header("Résumé Statistique des Données")
        st.write("Nombre de lignes et de colonnes :")
        st.write(df.shape)

        st.write("Noms des colonnes :")
        st.write(df.columns.tolist())

        st.write("Nombre de valeurs manquantes par colonne :")
        missing_values_count = df.isnull().sum()
        missing_values_percent = (missing_values_count / len(df)) * 100
        missing_values_table = pd.DataFrame({'Valeurs manquantes': missing_values_count, 'Pourcentage (%)': missing_values_percent})
        st.write(missing_values_table.T)  # Transposer pour un affichage horizontal

        st.write("Résumé statistique :")
        st.write(df.describe(include='all').T)  # Transposer pour un affichage horizontal

        # Analyse de la Distribution des Données
        analyze_data_distribution(df)

        # Prétraitement et Nettoyage des Données
        st.header("Prétraitement et Nettoyage des Données")

        # Options supplémentaires pour la suppression des lignes et colonnes
        st.subheader('Suppression de valeurs manquantes')
        if st.button('Supprimer les doublons'):
            st.session_state.df = remove_duplicates(st.session_state.df)
            st.write('Lignes doublons supprimées')
            st.write(st.session_state.df)

        if st.button('Supprimer les lignes entièrement vides'):
            st.session_state.df = remove_empty_rows(st.session_state.df)
            st.write('Lignes entièrement vides supprimées')
            st.write(st.session_state.df)

        if st.button('Supprimer les lignes avec plus de 80% de valeurs manquantes'):
            st.session_state.df = remove_high_missing_rows(st.session_state.df)
            st.write('Lignes avec plus de 80% de valeurs manquantes supprimées')
            st.write(st.session_state.df)

        if st.button('Supprimer les colonnes avec plus de 80% de valeurs manquantes'):
            st.session_state.df = remove_high_missing_cols(st.session_state.df)
            st.write('Colonnes avec plus de 80% de valeurs manquantes supprimées')
            st.write(st.session_state.df)

        # Créer une copie du DataFrame nettoyé pour la suite
        df_cleaned = st.session_state.df.copy()

        # Recalcul des valeurs manquantes après suppression
        st.write("Nombre de valeurs manquantes par colonne après suppression :")
        missing_values_count = df_cleaned.isnull().sum()
        missing_values_percent = (missing_values_count / len(df_cleaned)) * 100
        missing_values_table = pd.DataFrame({'Valeurs manquantes': missing_values_count, 'Pourcentage (%)': missing_values_percent})
        st.write(missing_values_table.T)  # Transposer pour un affichage horizontal

        # Gestion des valeurs manquantes
        st.subheader("Gestion des valeurs manquantes")
        missing_value_method = st.selectbox(
            "Sélectionnez une méthode pour gérer les valeurs manquantes",
            ("Aucune", "Remplacer par la moyenne", "Remplacer par la médiane", "Remplacer par le mode")
        )

        if missing_value_method != "Aucune":
            df_cleaned = handle_missing_values(df_cleaned, missing_value_method)

        st.write("Données après gestion des valeurs manquantes :")
        st.write(df_cleaned)

        # Normalisation des données
        st.subheader("Normalisation des données")
        normalization_method = st.selectbox(
            "Sélectionnez une méthode de normalisation",
            ("Aucune", "Min-Max", "Z-Score")
        )

        if normalization_method != "Aucune":
            df_cleaned = normalize_data(df_cleaned, normalization_method)

        st.write("Données après normalisation :")
        st.write(df_cleaned)

        # Visualisation des Données
        st.header("Visualisation des Données")
        column_to_plot = st.selectbox("Sélectionnez une colonne à visualiser", df_cleaned.columns)
        visualize_data(df_cleaned, column_to_plot)

        # Perform PCA
        perform_pca(df_cleaned)

        # Perform LDA
        perform_lda(df_cleaned)

        # Perform Clustering
        perform_clustering(df_cleaned)

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")

