
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Title
st.title("Heart Disease Patient Risk Analysis using K-Means")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Heart Dataset CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    # Dataset Shape
    st.write("Dataset Shape:", df.shape)

    # Column Names
    st.subheader("Columns")
    st.write(df.columns.tolist())

    # Remove Target Column
    if 'HeartDisease' in df.columns:
        X = df.drop('HeartDisease', axis=1)
    else:
        X = df.copy()

    # Convert categorical data
    X = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar
    st.sidebar.header("K-Means Settings")

    n_clusters = st.sidebar.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=2
    )

    # KMeans Model
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    # Add Cluster Column
    df['Cluster'] = clusters

    # Show Clustered Data
    st.subheader("Clustered Dataset")
    st.dataframe(df.head())

    # Inertia
    st.subheader("Inertia")
    st.write(kmeans.inertia_)

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=clusters
    )

    ax.set_title("K-Means Clusters")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    st.pyplot(fig)

    # Cluster Analysis
    st.subheader("Cluster Mean Analysis")

    numeric_df = df.select_dtypes(include=np.number)
    cluster_analysis = numeric_df.groupby('Cluster').mean()

    st.dataframe(cluster_analysis)

else:
    st.info("Please upload a CSV file.")
