"""
Nexus AI — Preprocessing Engine
-------------------------------
Modularized, robust transformations for cleaning, missing values, outliers,
encoding, feature engineering, dimensional reduction, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer, KNNImputer
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except ImportError:
    IterativeImputer = None

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import IsolationForest
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ============================================================
# 1. DUPLICATE HANDLING & DATA CLEANING
# ============================================================

def remove_duplicates(df: pd.DataFrame, subset: list = None, keep: str = "first") -> tuple:
    """
    Remove duplicate rows from dataframe.
    keep: 'first', 'last', False (drops all duplicates)
    Returns: (cleaned_df, count_removed)
    """
    initial_count = len(df)
    subset_cols = subset if subset else None
    cleaned_df = df.drop_duplicates(subset=subset_cols, keep=keep).copy()
    count_removed = initial_count - len(cleaned_df)
    return cleaned_df, count_removed


def drop_unwanted_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop specified columns safely."""
    valid_cols = [c for c in columns if c in df.columns]
    return df.drop(columns=valid_cols).copy()


def drop_constant_columns(df: pd.DataFrame) -> tuple:
    """Identify and drop single-value / constant columns."""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    cleaned_df = df.drop(columns=constant_cols).copy()
    return cleaned_df, constant_cols


# ============================================================
# 2. MISSING VALUES IMPUTATION
# ============================================================

def drop_missing_values(df: pd.DataFrame, axis: int = 0, how: str = "any", thresh: int = None, subset: list = None) -> pd.DataFrame:
    """Drop missing rows or columns."""
    return df.dropna(axis=axis, how=how, thresh=thresh, subset=subset).copy()


def impute_missing_values(
    df: pd.DataFrame,
    columns: list,
    strategy: str = "mean",
    fill_value: str = None,
    n_neighbors: int = 5,
    max_iter: int = 10
) -> pd.DataFrame:
    """
    Impute missing values using Simple, KNN, or Iterative (MICE) imputers.
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    if not valid_cols:
        return result

    strat = strategy.lower()

    if strat in ["mean", "median"]:
        # Only for numeric columns
        num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
        if num_cols:
            imputer = SimpleImputer(strategy=strat)
            result[num_cols] = imputer.fit_transform(result[num_cols])

    elif strat == "most_frequent" or strat == "mode":
        imputer = SimpleImputer(strategy="most_frequent")
        result[valid_cols] = imputer.fit_transform(result[valid_cols])

    elif strat == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value if fill_value is not None else "Missing")
        result[valid_cols] = imputer.fit_transform(result[valid_cols])

    elif strat == "knn":
        num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
        if num_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            result[num_cols] = imputer.fit_transform(result[num_cols])

    elif strat == "iterative" or strat == "mice":
        if IterativeImputer is not None:
            num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
            if num_cols:
                imputer = IterativeImputer(max_iter=max_iter, random_state=42)
                result[num_cols] = imputer.fit_transform(result[num_cols])
        else:
            # Fallback to mean
            num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
            if num_cols:
                imputer = SimpleImputer(strategy="mean")
                result[num_cols] = imputer.fit_transform(result[num_cols])

    return result


# ============================================================
# 3. OUTLIER DETECTION & TREATMENT
# ============================================================

def handle_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = "iqr",
    action: str = "clip",
    factor: float = 1.5,
    lower_pct: float = 0.05,
    upper_pct: float = 0.05,
    contamination: float = 0.05,
) -> tuple:
    """
    Detect and handle outliers using IQR, Z-Score, Winsorization, or Isolation Forest.
    action: 'clip' (capping), 'remove' (drop rows)
    Returns: (processed_df, outliers_count)
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols:
        return result, 0

    method = method.lower()
    action = action.lower()
    total_outliers = 0

    if method == "iqr":
        for col in valid_cols:
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)

            outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
            total_outliers += int(outlier_mask.sum())

            if action == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == "remove":
                result = result[~outlier_mask].copy()

    elif method == "zscore":
        for col in valid_cols:
            mean = result[col].mean()
            std = result[col].std()
            if std == 0 or np.isnan(std):
                continue
            z_scores = (result[col] - mean) / std
            outlier_mask = z_scores.abs() > factor
            total_outliers += int(outlier_mask.sum())

            lower_bound = mean - (factor * std)
            upper_bound = mean + (factor * std)

            if action == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == "remove":
                result = result[~outlier_mask].copy()

    elif method == "winsorize":
        for col in valid_cols:
            vals = result[col].dropna().values
            if len(vals) > 0:
                winsorized = winsorize(vals, limits=[lower_pct, upper_pct])
                result.loc[result[col].notnull(), col] = winsorized
                total_outliers += int((result[col] != vals).sum())

    elif method == "isolation_forest":
        iso = IsolationForest(contamination=contamination, random_state=42)
        clean_num = result[valid_cols].dropna()
        if len(clean_num) > 10:
            preds = iso.fit_predict(clean_num)
            outlier_indices = clean_num.index[preds == -1]
            total_outliers = len(outlier_indices)
            if action == "remove":
                result = result.drop(index=outlier_indices).copy()

    return result, total_outliers


# ============================================================
# 4. CATEGORICAL ENCODING
# ============================================================

def apply_categorical_encoding(df: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """
    Encode categorical columns based on strategy mapping:
    decisions = {col_name: 'one_hot' | 'label' | 'ordinal' | 'frequency'}
    """
    result = df.copy()

    for col, method in decisions.items():
        if col not in result.columns:
            continue

        method = str(method).lower()

        if method in ["one_hot", "one-hot", "ohe"]:
            # Max 30 categories to avoid explosion
            unique_count = result[col].nunique()
            if unique_count > 30:
                top_cats = result[col].value_counts().nlargest(29).index
                result[col] = result[col].apply(lambda x: x if x in top_cats else "Other")

            dummies = pd.get_dummies(result[col], prefix=col, drop_first=False, dtype=int)
            result = pd.concat([result.drop(columns=[col]), dummies], axis=1)

        elif method == "label":
            le = LabelEncoder()
            # Handle potential NaNs
            filled_col = result[col].fillna("Missing").astype(str)
            result[col] = le.fit_transform(filled_col)

        elif method == "ordinal":
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            filled_col = result[[col]].fillna("Missing").astype(str)
            result[col] = oe.fit_transform(filled_col).ravel()

        elif method == "frequency":
            freq_map = result[col].value_counts(normalize=True).to_dict()
            result[col] = result[col].map(freq_map).fillna(0)

    return result


# ============================================================
# 5. FEATURE TRANSFORMATIONS & POLYNOMIAL
# ============================================================

def transform_features(df: pd.DataFrame, columns: list, method: str = "log1p") -> pd.DataFrame:
    """
    Apply mathematical transformations: 'log1p', 'box-cox', 'yeo-johnson'.
    Automatically shifts non-positive numbers for log and Box-Cox.
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols:
        return result

    method = method.lower()

    if method in ["log", "log1p"]:
        for col in valid_cols:
            min_val = result[col].min()
            if pd.isna(min_val):
                continue
            if min_val <= 0:
                shift = abs(min_val) + 1.0
                result[col] = np.log1p(result[col] + shift)
            else:
                result[col] = np.log1p(result[col])

    elif method == "box-cox":
        for col in valid_cols:
            values = result[col].to_numpy(dtype=float)
            min_val = np.nanmin(values)
            if min_val <= 0:
                values = values + abs(min_val) + 1.0
            pt = PowerTransformer(method="box-cox", standardize=True)
            result[col] = pt.fit_transform(values.reshape(-1, 1)).ravel()

    elif method == "yeo-johnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        result[valid_cols] = pt.fit_transform(result[valid_cols])

    return result


def generate_polynomial_features(
    df: pd.DataFrame,
    columns: list,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False
) -> pd.DataFrame:
    """Generate polynomial and interaction terms for selected features."""
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols or degree < 1:
        return result

    # Restrict large expansions
    if len(valid_cols) > 15 and degree > 2:
        raise ValueError("Too many columns selected for degree > 2 expansion.")

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    poly_data = poly.fit_transform(result[valid_cols])
    poly_feature_names = poly.get_feature_names_out(valid_cols)

    poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=result.index)
    result = pd.concat([result.drop(columns=valid_cols), poly_df], axis=1)
    return result


# ============================================================
# 6. DIMENSIONALITY REDUCTION (PCA) & RFE
# ============================================================

def apply_pca(df: pd.DataFrame, columns: list, n_components=0.95, standardize: bool = True) -> tuple:
    """
    Apply PCA to numerical features.
    Returns: (pca_df, pca_model, explained_variance_ratio)
    """
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        raise ValueError("PCA requires numerical features.")

    X = df[valid_cols].copy()
    if standardize:
        scaler = StandardScaler()
        X_mat = scaler.fit_transform(X)
    else:
        X_mat = X.to_numpy()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_mat)
    pc_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pc_names, index=df.index)

    # Retain non-PCA columns if any
    other_cols = [c for c in df.columns if c not in valid_cols]
    if other_cols:
        pca_df = pd.concat([df[other_cols], pca_df], axis=1)

    return pca_df, pca, pca.explained_variance_ratio_


def apply_rfe_selection(X: pd.DataFrame, y: pd.Series, n_features: int = 5, problem_type: str = "Classification") -> tuple:
    """
    Apply Recursive Feature Elimination (RFE).
    Returns: (X_selected, selected_column_names, rfe_model)
    """
    if len(X) != len(y):
        raise ValueError("X and y sample lengths must match.")

    n_features = max(1, min(n_features, X.shape[1]))

    if problem_type == "Classification":
        estimator = LogisticRegression(max_iter=1500, random_state=42)
    else:
        estimator = LinearRegression()

    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)

    selected_cols = X.columns[rfe.support_].tolist()
    X_selected = X[selected_cols].copy()
    return X_selected, selected_cols, rfe


# ============================================================
# 7. CLASS IMBALANCE HANDLING
# ============================================================

def handle_imbalanced_classes(X_train: pd.DataFrame, y_train: pd.Series, method: str = "smote") -> tuple:
    """
    Balance classification dataset using SMOTE or Random Under-sampling.
    """
    method = method.lower()
    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Dataset requires at least 2 distinct classes.")

    if method == "smote":
        min_samples = class_counts.min()
        k_neighbors = min(5, max(1, min_samples - 1))
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    elif method == "undersampling":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Method must be 'smote' or 'undersampling'.")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# ============================================================
# 8. FEATURE SCALING
# ============================================================

def apply_feature_scaling(df: pd.DataFrame, columns: list, scaler_type: str = "standard") -> pd.DataFrame:
    """
    Scale numeric features using Standard, MinMax, Robust, MaxAbs, or Normalizer.
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols:
        return result

    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type in ["minmax", "min-max"]:
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler()
    elif scaler_type == "normalizer":
        scaler = Normalizer()
    else:
        scaler = StandardScaler()

    result[valid_cols] = scaler.fit_transform(result[valid_cols])
    return result
