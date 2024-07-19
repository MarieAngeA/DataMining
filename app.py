import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from io import StringIO

# CSS pour le fond bleu clair pastel
st.markdown("""
    <style>
        .stApp {
            background-color: #e0f7fa;
        }
        .main-header {
            color: #00695c;
        }
        .subheader {
            color: #004d40;
        }
    </style>
    """, unsafe_allow_html=True)

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

    calculate_correlation(data)

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
        
        st.write("Composants PCA:")
        st.write(pd.DataFrame(pca.components_, columns=data.select_dtypes(include=['float64', 'int64']).columns, index=['PCA1', 'PCA2']))
        
    except Exception as e:
        st.error(f"Erreur lors du calcul du PCA : {e}")

def perform_lda(data):
    st.header("Analyse Discriminante Linéaire (LDA)")
    if 'Cluster' in data.columns:
        try:
            lda = LDA(n_components=2)
            clean_data = data.dropna(subset=['Cluster'])
            lda_components = lda.fit_transform(clean_data.select_dtypes(include=['float64', 'int64']), clean_data['Cluster'])
            lda_df = pd.DataFrame(data=lda_components, columns=['LDA1', 'LDA2'])
            st.write(lda_df)

            fig, ax = plt.subplots()
            scatter = ax.scatter(lda_df['LDA1'], lda_df['LDA2'], c=clean_data['Cluster'], cmap='viridis')
            ax.set_xlabel('LDA1')
            ax.set_ylabel('LDA2')
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors du calcul du LDA : {e}")
    else:
        st.write("L'Analyse Discriminante Linéaire (LDA) nécessite des clusters. Veuillez effectuer le clustering d'abord.")

def plot_elbow_method(data):
    st.header("Détermination du Nombre Optimal de Clusters (Méthode du Coude)")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title('Méthode du Coude')
    ax.set_xlabel('Nombre de Clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

def perform_clustering(data, key_suffix=""):
    st.header("Clustering des Données")

    plot_elbow_method(data.select_dtypes(include=['float64', 'int64']).dropna())

    clustering_method = st.selectbox(
        "Sélectionnez une méthode de clustering",
        ("K-Means", "DBSCAN"),
        key=f"clustering_method{key_suffix}"
    )

    clean_data = data.select_dtypes(include=['float64', 'int64']).dropna()

    if clustering_method == "K-Means":
        num_clusters = st.slider("Nombre de clusters (K)", 2, 10, 3, key=f"num_clusters{key_suffix}")
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(clean_data)
        score = kmeans.inertia_
    elif clustering_method == "DBSCAN":
        eps = st.slider("EPS", 0.1, 10.0, 0.5, key=f"eps{key_suffix}")
        min_samples = st.slider("Nombre minimum d'échantillons", 1, 10, 5, key=f"min_samples{key_suffix}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(clean_data)
        score = len(set(clusters)) - (1 if -1 in clusters else 0)

    cluster_column = pd.Series(index=clean_data.index, data=clusters)
    data['Cluster'] = cluster_column

    st.write("Aperçu des clusters :")
    st.write(data)

    st.header("Visualisation des Clusters")
    try:
        pca = PCA(n_components=2)
        components = pca.fit_transform(clean_data)

        fig, ax = plt.subplots()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=cluster_column, cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        st.pyplot(fig)

        st.write("Statistiques des clusters :")
        st.write(data.groupby('Cluster').size().reset_index(name='Count'))
    except Exception as e:
        st.error(f"Erreur lors de la visualisation des clusters : {e}")

    return score

def calculate_correlation(data):
    st.header("Matrice de Corrélation")
    corr_matrix = data.corr()
    st.write(corr_matrix)

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.title('Application d\'Exploration et de Clustering des Données')


menu = st.sidebar.radio("Menu", ["Téléchargement du fichier", "Aperçu des Données", "Résumé Statistique des Données", "Prétraitement et Nettoyage des Données", "Visualisation des Données", "Analyse en Composantes Principales (PCA)", "Analyse Discriminante Linéaire (LDA)", "Clustering des Données"])

if menu == "Téléchargement du fichier":
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

    header_option = st.selectbox("Ligne d'entête", [None, 0, 1, 2, 3], index=1, key="header_option")
    separator_option = st.selectbox("Type de séparateur", [",", ";", "\t", "|"], index=0, key="separator_option")

    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    encoding_option = st.selectbox("Choisissez un encodage", encodings, index=0, key="encoding_option")

    if uploaded_file is not None:
        try:

            bytes_data = uploaded_file.getvalue()

            stringio = StringIO(bytes_data.decode(encoding_option))

            string_data = stringio.read()

            try:
                df = pd.read_csv(StringIO(string_data), header=header_option, delimiter=separator_option)
            except pd.errors.ParserError as e:
                st.error(f"Erreur de parsing du fichier : {e}")
                st.stop()
            except UnicodeDecodeError as e:
                st.error(f"Erreur d'encodage du fichier : {e}")
                st.stop()

            st.session_state.df = df
            st.session_state.df_cleaned = df.copy()  

            st.write("DataFrame :", df.head())

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

if 'df' in st.session_state:
    df = st.session_state.df
    df_cleaned = st.session_state.df_cleaned

    if menu == "Aperçu des Données":
        st.header("Aperçu des Données")
        st.write("Premières lignes du dataset :")
        st.write(df.head())

        st.write("Dernières lignes du dataset :")
        st.write(df.tail())

    elif menu == "Résumé Statistique des Données":
        st.header("Résumé Statistique des Données")
        st.write("Nombre de lignes et de colonnes :")
        st.write(df.shape)

        st.write("Noms des colonnes :")
        st.write(df.columns.tolist())

        st.write("Nombre de valeurs manquantes par colonne :")
        missing_values_count = df.isnull().sum()
        missing_values_percent = (missing_values_count / len(df)) * 100
        missing_values_table = pd.DataFrame({'Valeurs manquantes': missing_values_count, 'Pourcentage (%)': missing_values_percent})
        st.write(missing_values_table.T) 

        st.write("Résumé statistique :")
        st.write(df.describe(include='all').T)  

    elif menu == "Prétraitement et Nettoyage des Données":
        st.header("Prétraitement et Nettoyage des Données")

        st.subheader('Suppression de valeurs manquantes')
        if st.button('Supprimer les doublons'):
            st.session_state.df_cleaned = remove_duplicates(st.session_state.df_cleaned)
            st.write('Lignes doublons supprimées')
            st.write(st.session_state.df_cleaned)

        if st.button('Supprimer les lignes entièrement vides'):
            st.session_state.df_cleaned = remove_empty_rows(st.session_state.df_cleaned)
            st.write('Lignes entièrement vides supprimées')
            st.write(st.session_state.df_cleaned)

        if st.button('Supprimer les lignes avec plus de 80% de valeurs manquantes'):
            st.session_state.df_cleaned = remove_high_missing_rows(st.session_state.df_cleaned)
            st.write('Lignes avec plus de 80% de valeurs manquantes supprimées')
            st.write(st.session_state.df_cleaned)

        if st.button('Supprimer les colonnes avec plus de 80% de valeurs manquantes'):
            st.session_state.df_cleaned = remove_high_missing_cols(st.session_state.df_cleaned)
            st.write('Colonnes avec plus de 80% de valeurs manquantes supprimées')
            st.write(st.session_state.df_cleaned)

        st.write("Nombre de valeurs manquantes par colonne après suppression :")
        missing_values_count = st.session_state.df_cleaned.isnull().sum()
        missing_values_percent = (missing_values_count / len(st.session_state.df_cleaned)) * 100
        missing_values_table = pd.DataFrame({'Valeurs manquantes': missing_values_count, 'Pourcentage (%)': missing_values_percent})
        st.write(missing_values_table.T)  

        st.subheader("Gestion des valeurs manquantes")
        missing_value_method = st.selectbox(
            "Sélectionnez une méthode pour gérer les valeurs manquantes",
            ("Aucune", "Remplacer par la moyenne", "Remplacer par la médiane", "Remplacer par le mode"),
            key="missing_value_method"
        )

        if missing_value_method != "Aucune":
            st.session_state.df_cleaned = handle_missing_values(st.session_state.df_cleaned, missing_value_method)

        st.write("Données après gestion des valeurs manquantes :")
        st.write(st.session_state.df_cleaned)

        st.subheader("Normalisation des données")
        normalization_method = st.selectbox(
            "Sélectionnez une méthode de normalisation",
            ("Aucune", "Min-Max", "Z-Score"),
            key="normalization_method"
        )

        if normalization_method != "Aucune":
            st.session_state.df_cleaned = normalize_data(st.session_state.df_cleaned, normalization_method)

        st.write("Données après normalisation :")
        st.write(st.session_state.df_cleaned)

    elif menu == "Visualisation des Données":
        st.header("Visualisation des Données")
        column_to_plot = st.selectbox("Sélectionnez une colonne à visualiser", df.columns, key="column_to_plot")
        visualize_data(df, column_to_plot)

    elif menu == "Analyse en Composantes Principales (PCA)":
        perform_pca(st.session_state.df_cleaned)

    elif menu == "Analyse Discriminante Linéaire (LDA)":
        perform_lda(st.session_state.df_cleaned)

    elif menu == "Clustering des Données":
        perform_clustering(st.session_state.df_cleaned, key_suffix="main")
