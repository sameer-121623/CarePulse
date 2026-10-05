import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(X_train_scaled: pd.DataFrame, n_components: int = 2) -> dict:
    """Apply PCA to reduce features to principal components.
    
    Reduces the high-dimensional feature space (numerical + one-hot encoded)
    into 2 principal components for visualization.
    
    Args:
        X_train_scaled: Scaled and encoded training feature matrix.
        n_components: Number of principal components (default: 2).
        
    Returns:
        Dictionary with:
        - 'components': DataFrame with PC1 and PC2 columns
        - 'explained_variance_ratio': Array of variance explained per component
        - 'total_explained_variance': Sum of explained variance ratios
        - 'pca_model': Fitted PCA object (for transforming test data if needed)
    """
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_train_scaled)
    
    component_names = [f'PC{i+1}' for i in range(n_components)]
    components_df = pd.DataFrame(components, columns=component_names)
    
    return {
        'components': components_df,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_explained_variance': sum(pca.explained_variance_ratio_),
        'pca_model': pca
    }
