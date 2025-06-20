import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_clusters(X, clusters, features):
    if len(features) >= 2:
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 1],
            color=clusters,
            labels={'x': features[0], 'y': features[1]},
            title='Cluster Visualization'
        )
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=clusters,
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
            title='Cluster Visualization (PCA)'
        )
    return fig

def check_class_imbalance(y):
    """Check if dataset is imbalanced based on class distribution"""
    class_counts = np.bincount(y)
    major_class_count = np.max(class_counts)
    minor_class_count = np.min(class_counts)
    return (major_class_count / minor_class_count) > 3


def plot_clusters_with_hulls(X_scaled_pca, clusters, optimal_k, kmeans_object):  # Corrected parameter
    plt.figure(figsize=(8, 6))

    for i in range(optimal_k):
        cluster_data = X_scaled_pca[X_scaled_pca['cluster'] == i] # Use X_scaled_pca
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {i}')

        if len(cluster_data) > 2:
            points = cluster_data[['PC1', 'PC2']].values
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1)

    plt.scatter(kmeans_object.cluster_centers_[:, 0], kmeans_object.cluster_centers_[:, 1], marker='x', s=200, c='black', label='Centroids')
    plt.title('K-means Clustering (2D PCA Visualization)')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt) 



def classify():
    st.write("# Classification & Clustering")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset in the Home section.")
        return
    
    df = st.session_state.df.copy()
    
    st.write("## Data Preparation")
    st.write("### Feature Selection")

    select_all = st.checkbox("Select all Features")

    if select_all:
        features = df.columns.tolist()
    else:
        features = st.multiselect("Select Feature Columns", df.columns)
    
    # target = st.selectbox("Select Target Column", df.columns)
    # features = st.multiselect("Select Feature Columns", df.columns)
    
    if not features:
        st.warning("⚠️ Please select at least one feature column.")
        return
    
    X = df[features]
    # y = df[target]
    
    # Handle categorical features (same for both KMeans and RF)
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(exclude=['object']).columns

    transformers = []
    if numeric_columns.any():
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformers.append(('num', numeric_transformer, numeric_columns))

    if categorical_columns.any():
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        transformers.append(('cat', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)

    transformed_feature_names = preprocessor.get_feature_names_out(input_features=X.columns)
    X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_feature_names)


    # Scale the data and create a new DataFrame for the scaled data (Approach 2)
    # X_scaled = preprocessor.fit_transform(X) #Fit and transform the data
    # X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

  # Model Selection
    st.write("## Model Configuration")
    model_choice = st.selectbox("Choose a model", ["K-Means Clustering (Clustering)", "Random Forest (New prediction)"], index=0) # KMeans default
    optimal_k = 3  # Or let user select

    if model_choice == "K-Means Clustering (Clustering)":
        st.write("## K-Means Configuration")

        with st.spinner("Performing clustering analysis..."):
            kmeans_pipeline = Pipeline([
                ('kmeans', KMeans(n_clusters=optimal_k, random_state=42, n_init=50, init='k-means++', tol=1e-4, max_iter=300)) #Removed elkan as it is not suitable for sparse matrix
            ])

            X_clustered = kmeans_pipeline.fit_transform(X_transformed_df) #Fit on the transformed data
            kmeans = kmeans_pipeline.named_steps['kmeans']
            clusters = kmeans.labels_

            df["Cluster"] = clusters
            st.write("### Cluster Assignments")
            st.write(df[["Cluster"]].head(10))

            n_components = 2
            pca = PCA(n_components=n_components)
            X_pca_data = pca.fit_transform(X_transformed_df)  # Fit PCA on the *transformed* data
            X_pca = pd.DataFrame(X_pca_data, columns=['PC1', 'PC2'])
            X_pca['cluster'] = clusters

            st.write("### Cluster Visualization")
            plot_clusters_with_hulls(X_pca, clusters, optimal_k, kmeans)  # Pass the PCA-transformed data

    elif model_choice == "Random Forest (New prediction)":  # Classification after KMeans
        st.write("## Random Forest Classification (using KMeans Clusters)")

        # 1. Perform KMeans Clustering 
        # optimal_k = 3  # Or let the user choose
        kmeans_pipeline = Pipeline([
            ('preprocessor', preprocessor),  # Use same preprocessor as before
            ('kmeans', KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++'))
        ])
        X_clustered = kmeans_pipeline.fit_transform(X)  # Fit and transform on original features

        kmeans = kmeans_pipeline.named_steps['kmeans']
        clusters = kmeans.labels_

        # 2. Add cluster labels as a new feature
        X['Cluster'] = clusters  # Add to the original X
        
        # 3. Prepare data for classification (EXCLUDING the 'Cluster' feature)
        X_class = X.drop('Cluster', axis=1)  # Remove the 'Cluster' column
        y_class = X['Cluster']

        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class,  # Features (excluding 'Cluster') 
                                                                                    y_class,  # Target variable ('Cluster' column)
                                                                                    test_size=0.2, random_state=42)

        # 4. Classification Model Training (Random Forest)
        with st.spinner("Training Random Forest Classifier..."):

            rf_pipeline = Pipeline([  # No need for SMOTE or class imbalance check
                ('classifier', RandomForestClassifier(random_state=42))
            ])

            param_grid = {  # Example grid, adjust as needed
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }

            grid_search_class = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1)
            grid_search_class.fit(X_train_class, y_train_class) # Train on Cluster labels
            rf_pipeline = grid_search_class.best_estimator_

        # --- New Point Classification ---
        st.write("### Classify New Data Point (using trained RF)")

        new_data = {}
        for feature in X_class.columns:  # Iterate over all features, excluding 'Cluster'
            if X_class[feature].dtype == 'object':  # Categorical feature
                unique_values = X_class[feature].unique()
                new_data[feature] = st.selectbox(f"Enter value for {feature}", unique_values)
            else:  # Numerical feature
                new_data[feature] = st.number_input(f"Enter value for {feature}", min_value=1, max_value=5, step=1)

        new_df = pd.DataFrame([new_data])

        if st.button("Predict Cluster"): # Changed button text
            try:
                new_df_transformed = preprocessor.transform(new_df)  # Use the preprocessor
                prediction = rf_pipeline.predict(new_df_transformed)  # Predict on the transformed data
                predicted_cluster = prediction[0]
                st.write(f"The predicted cluster is: {predicted_cluster}")

            except ValueError as e:  # Catch potential errors
                st.error(f"Error during prediction: {e}")
                st.write("Check if all features are entered correctly and in the expected format.")
    