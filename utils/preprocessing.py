"""
Nexus AI — Preprocessing Engine & Pipeline Manager
--------------------------------------------------
Modularized, robust transformations for cleaning, missing values, outliers,
encoding, feature engineering, dimensional reduction, and scaling.
Includes PreprocessingPipeline class for deterministic raw-data inference.
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
# 1. STATEFUL PREPROCESSING PIPELINE CLASS
# ============================================================

class PreprocessingPipeline:
    """
    Stateful Preprocessing Pipeline that stores every transformation step,
    fitted encoder, scaler, imputer, and feature alignment rule applied during training.
    Enables automatic, deterministic transformation of RAW user inputs for prediction.
    """
    def __init__(self):
        self.raw_columns = []
        self.raw_dtypes = {}
        self.raw_categorical_values = {}
        self.raw_numeric_stats = {}
        self.target_column = None
        self.problem_type = "Classification"
        self.target_encoder = None
        self.target_classes_ = None

        # Transformation Registry
        self.dropped_columns = []
        self.imputers = {}             # {col: {'strategy': str, 'imputer': object, 'fill_value': any}}
        self.global_imputer_info = None
        self.outlier_bounds = {}       # {col: (lower_bound, upper_bound)}
        self.encoding_decisions = {}   # {col: 'one_hot'|'label'|'ordinal'|'frequency'}
        self.encoders = {}             # {col: {'type': ..., 'obj': ..., 'categories': ..., 'mapping': ..., 'dummy_cols': [...]}}
        self.math_transforms = {}      # {col: {'method': ..., 'shift': ..., 'transformer': ...}}
        self.polynomial_transformer = None
        self.poly_columns = []
        self.pca_transformer = None
        self.pca_scaler = None
        self.pca_columns = []
        self.scaler = None
        self.scaler_type = None
        self.scaled_columns = []
        self.selected_features = []
        self.final_model_features = []
        self.applied_stages = []

    def init_from_raw(self, raw_df: pd.DataFrame, target_col: str = None, problem_type: str = "Classification"):
        """Initialize schema metadata from raw original dataset."""
        if raw_df is None or raw_df.empty:
            return

        self.raw_columns = list(raw_df.columns)
        self.target_column = target_col
        self.problem_type = problem_type
        self.raw_dtypes = {col: str(raw_df[col].dtype) for col in raw_df.columns}
        self.raw_categorical_values = {}
        self.raw_numeric_stats = {}

        for col in raw_df.columns:
            if pd.api.types.is_numeric_dtype(raw_df[col]):
                vals = raw_df[col].dropna()
                if len(vals) > 0:
                    self.raw_numeric_stats[col] = {
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "std": float(vals.std()) if len(vals) > 1 else 1.0,
                    }
                else:
                    self.raw_numeric_stats[col] = {"min": 0.0, "max": 1.0, "mean": 0.0, "median": 0.0, "std": 1.0}
            else:
                uniques = [str(x) for x in raw_df[col].dropna().unique() if pd.notna(x)]
                self.raw_categorical_values[col] = sorted(uniques)

    def record_dropped_columns(self, cols: list):
        """Record columns dropped during data cleaning."""
        for c in cols:
            if c not in self.dropped_columns:
                self.dropped_columns.append(c)
        if "Data Cleaning" not in self.applied_stages:
            self.applied_stages.append("Data Cleaning")

    def record_imputation(self, strategy: str, columns: list, imputers_dict: dict = None):
        """Record fitted imputer parameters."""
        if imputers_dict:
            self.imputers.update(imputers_dict)
        self.global_imputer_info = {"strategy": strategy, "columns": columns}
        if "Missing Imputation" not in self.applied_stages:
            self.applied_stages.append("Missing Imputation")

    def record_outlier_bounds(self, bounds_dict: dict):
        """Record outlier boundary thresholds for clipping."""
        self.outlier_bounds.update(bounds_dict)
        if "Outlier Treatment" not in self.applied_stages:
            self.applied_stages.append("Outlier Treatment")

    def record_encoding(self, decisions: dict, encoders_dict: dict):
        """Record fitted categorical encoders and dummy structure."""
        self.encoding_decisions.update(decisions)
        self.encoders.update(encoders_dict)
        if "Categorical Encoding" not in self.applied_stages:
            self.applied_stages.append("Categorical Encoding")

    def record_math_transforms(self, method: str, columns: list, transforms_dict: dict):
        """Record mathematical transformations and shifts."""
        self.math_transforms.update(transforms_dict)
        if "Math Transformation" not in self.applied_stages:
            self.applied_stages.append("Math Transformation")

    def record_polynomial(self, poly_transformer: PolynomialFeatures, columns: list):
        """Record polynomial expansion transformer."""
        self.polynomial_transformer = poly_transformer
        self.poly_columns = list(columns)
        if "Polynomial Features" not in self.applied_stages:
            self.applied_stages.append("Polynomial Features")

    def record_pca(self, pca_transformer: PCA, pca_scaler: StandardScaler, columns: list):
        """Record PCA transformer and standardizer."""
        self.pca_transformer = pca_transformer
        self.pca_scaler = pca_scaler
        self.pca_columns = list(columns)
        if "PCA Reduction" not in self.applied_stages:
            self.applied_stages.append("PCA Reduction")

    def record_scaling(self, scaler_obj, scaler_type: str, columns: list):
        """Record fitted feature scaler."""
        self.scaler = scaler_obj
        self.scaler_type = scaler_type
        self.scaled_columns = list(columns)
        if "Feature Scaling" not in self.applied_stages:
            self.applied_stages.append("Feature Scaling")

    def record_model_features(self, selected_features: list, target_col: str = None, problem_type: str = None):
        """Finalize the exact model features and ordering."""
        self.selected_features = list(selected_features)
        self.final_model_features = list(selected_features)
        if target_col is not None:
            self.target_column = target_col
        if problem_type is not None:
            self.problem_type = problem_type

    def get_required_raw_features(self) -> list:
        """Return the list of raw features expected from the user for prediction."""
        if not self.raw_columns:
            return self.final_model_features

        req = []
        for col in self.raw_columns:
            if col == self.target_column:
                continue
            if col in self.dropped_columns:
                continue
            req.append(col)
        return req if req else self.final_model_features

    def validate_raw_input(self, raw_df: pd.DataFrame, require_target: bool = False) -> tuple:
        """
        Validate incoming raw test dataframe against training schema.
        Returns: (is_valid: bool, errors: list[str], warnings: list[str])
        """
        errors = []
        warnings = []

        if raw_df is None or raw_df.empty:
            return False, ["Input dataset is empty. Please provide valid data rows."], []

        required_features = self.get_required_raw_features()
        missing_features = [f for f in required_features if f not in raw_df.columns]

        if missing_features:
            errors.append(f"Missing required feature columns: {', '.join(missing_features)}")

        if require_target and self.target_column and self.target_column not in raw_df.columns:
            errors.append(f"Target column '{self.target_column}' is missing from the test dataset.")

        extra_cols = [c for c in raw_df.columns if c not in self.raw_columns and c not in required_features and c != self.target_column]
        if extra_cols:
            warnings.append(f"Ignored {len(extra_cols)} extra columns not present in training data: {', '.join(extra_cols[:5])}")

        # Check numeric conversions
        for col in required_features:
            if col in raw_df.columns and col in self.raw_numeric_stats:
                if not pd.api.types.is_numeric_dtype(raw_df[col]):
                    converted = pd.to_numeric(raw_df[col], errors="coerce")
                    if converted.isnull().all() and len(raw_df) > 0:
                        errors.append(f"Column '{col}' expects numerical values, but received invalid data types.")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full end-to-end preprocessing pipeline on RAW user data.
        Applies ONLY .transform() methods without re-fitting.
        """
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        df = raw_df.copy()

        # Step 1: Drop explicitly dropped training columns
        for col in self.dropped_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Step 2: Imputation
        if self.imputers:
            for col, imp_info in self.imputers.items():
                if col in df.columns:
                    imputer = imp_info.get("imputer")
                    if imputer is not None and hasattr(imputer, "transform"):
                        try:
                            df[[col]] = imputer.transform(df[[col]])
                        except Exception:
                            fill_val = imp_info.get("fill_value", 0)
                            df[col] = df[col].fillna(fill_val)
                    elif imp_info.get("fill_value") is not None:
                        df[col] = df[col].fillna(imp_info["fill_value"])

        # Fallback imputation for remaining nulls based on training statistics
        for col in df.columns:
            if df[col].isnull().any():
                if col in self.raw_numeric_stats:
                    df[col] = df[col].fillna(self.raw_numeric_stats[col]["median"])
                elif col in self.raw_categorical_values and self.raw_categorical_values[col]:
                    df[col] = df[col].fillna(self.raw_categorical_values[col][0])
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna("Missing")

        # Step 3: Outlier Clipping
        if self.outlier_bounds:
            for col, (low, high) in self.outlier_bounds.items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].clip(lower=low, upper=high)

        # Step 4: Categorical Encoding
        if self.encoders:
            for col, enc_data in self.encoders.items():
                if col not in df.columns:
                    continue

                enc_type = enc_data.get("type", "one_hot")

                if enc_type in ["one_hot", "one-hot", "ohe"]:
                    dummy_cols = enc_data.get("dummy_cols", [])
                    col_series = df[col].astype(str)
                    for d_col in dummy_cols:
                        cat_name = enc_data.get("dummy_to_cat", {}).get(d_col)
                        if cat_name is not None:
                            df[d_col] = (col_series == cat_name).astype(int)
                        else:
                            # Suffix match fallback
                            suffix = d_col[len(col) + 1:] if d_col.startswith(col + "_") else d_col
                            df[d_col] = (col_series == suffix).astype(int)
                    df = df.drop(columns=[col])

                elif enc_type in ["ordinal", "label"]:
                    encoder_obj = enc_data.get("obj")
                    if encoder_obj is not None and hasattr(encoder_obj, "transform"):
                        try:
                            df[col] = encoder_obj.transform(df[[col]].astype(str)).ravel()
                        except Exception:
                            # Map lookup
                            mapping = enc_data.get("mapping", {})
                            df[col] = df[col].astype(str).map(mapping).fillna(-1)
                    else:
                        mapping = enc_data.get("mapping", {})
                        df[col] = df[col].astype(str).map(mapping).fillna(-1)

                elif enc_type == "frequency":
                    mapping = enc_data.get("mapping", {})
                    df[col] = df[col].map(mapping).fillna(0.0)

        # Step 5: Mathematical Transformations
        if self.math_transforms:
            for col, t_info in self.math_transforms.items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    method = t_info.get("method")
                    shift = t_info.get("shift", 0.0)
                    transformer = t_info.get("transformer")

                    if method == "log1p":
                        df[col] = np.log1p(df[col] + shift if shift > 0 else df[col])
                    elif transformer is not None and hasattr(transformer, "transform"):
                        try:
                            vals = df[[col]].to_numpy(dtype=float)
                            if shift > 0:
                                vals = vals + shift
                            df[col] = transformer.transform(vals).ravel()
                        except Exception:
                            pass

        # Step 6: Polynomial Expansion
        if self.polynomial_transformer is not None and self.poly_columns:
            valid_poly = [c for c in self.poly_columns if c in df.columns]
            if len(valid_poly) == len(self.poly_columns):
                try:
                    poly_arr = self.polynomial_transformer.transform(df[valid_poly])
                    poly_names = self.polynomial_transformer.get_feature_names_out(valid_poly)
                    poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
                    df = pd.concat([df.drop(columns=valid_poly), poly_df], axis=1)
                except Exception:
                    pass

        # Step 7: PCA Dimensionality Reduction
        if self.pca_transformer is not None and self.pca_columns:
            valid_pca = [c for c in self.pca_columns if c in df.columns]
            if len(valid_pca) == len(self.pca_columns):
                try:
                    X_pca_in = df[valid_pca].copy()
                    if self.pca_scaler is not None:
                        X_pca_in = self.pca_scaler.transform(X_pca_in)
                    pca_arr = self.pca_transformer.transform(X_pca_in)
                    pc_names = [f"PC_{i+1}" for i in range(pca_arr.shape[1])]
                    pca_df = pd.DataFrame(pca_arr, columns=pc_names, index=df.index)
                    other_cols = [c for c in df.columns if c not in valid_pca]
                    df = pd.concat([df[other_cols], pca_df], axis=1) if other_cols else pca_df
                except Exception:
                    pass

        # Step 8: Feature Scaling
        if self.scaler is not None and self.scaled_columns:
            valid_scale = [c for c in self.scaled_columns if c in df.columns]
            if valid_scale:
                try:
                    df[valid_scale] = self.scaler.transform(df[valid_scale])
                except Exception:
                    pass

        # Step 9: Align exact Model Features in strict order
        if self.final_model_features:
            for feat in self.final_model_features:
                if feat not in df.columns:
                    df[feat] = 0.0
            df = df[self.final_model_features].copy()

        return df

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        """Decode model prediction array back to original natural class labels."""
        if self.target_encoder is not None and hasattr(self.target_encoder, "inverse_transform"):
            try:
                y_arr = np.asarray(y_pred).round().astype(int) if np.issubdtype(np.asarray(y_pred).dtype, np.floating) else np.asarray(y_pred)
                return self.target_encoder.inverse_transform(y_arr)
            except Exception:
                pass

        if self.target_classes_ is not None:
            try:
                y_arr = np.asarray(y_pred).round().astype(int)
                if all(0 <= i < len(self.target_classes_) for i in y_arr):
                    return np.array([self.target_classes_[i] for i in y_arr])
            except Exception:
                pass

        return y_pred


# ============================================================
# 2. DUPLICATE HANDLING & DATA CLEANING
# ============================================================

def remove_duplicates(df: pd.DataFrame, subset: list = None, keep: str = "first") -> tuple:
    """Remove duplicate rows from dataframe."""
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
# 3. MISSING VALUES IMPUTATION
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
) -> tuple:
    """
    Impute missing values and return (result_df, fitted_imputers_dict).
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    if not valid_cols:
        return result, {}

    strat = strategy.lower()
    imputers_dict = {}

    if strat in ["mean", "median"]:
        num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
        for col in num_cols:
            imputer = SimpleImputer(strategy=strat)
            result[[col]] = imputer.fit_transform(result[[col]])
            imputers_dict[col] = {"strategy": strat, "imputer": imputer, "fill_value": float(imputer.statistics_[0])}

    elif strat in ["most_frequent", "mode"]:
        for col in valid_cols:
            imputer = SimpleImputer(strategy="most_frequent")
            result[[col]] = imputer.fit_transform(result[[col]])
            imputers_dict[col] = {"strategy": "most_frequent", "imputer": imputer, "fill_value": str(imputer.statistics_[0])}

    elif strat == "constant":
        const_val = fill_value if fill_value is not None else "Missing"
        for col in valid_cols:
            imputer = SimpleImputer(strategy="constant", fill_value=const_val)
            result[[col]] = imputer.fit_transform(result[[col]])
            imputers_dict[col] = {"strategy": "constant", "imputer": imputer, "fill_value": const_val}

    elif strat == "knn":
        num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
        for col in num_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            result[[col]] = imputer.fit_transform(result[[col]])
            imputers_dict[col] = {"strategy": "knn", "imputer": imputer, "fill_value": float(result[col].mean())}

    elif strat in ["iterative", "mice"]:
        num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(result[c])]
        for col in num_cols:
            if IterativeImputer is not None:
                imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            else:
                imputer = SimpleImputer(strategy="mean")
            result[[col]] = imputer.fit_transform(result[[col]])
            imputers_dict[col] = {"strategy": "iterative", "imputer": imputer, "fill_value": float(result[col].mean())}

    return result, imputers_dict


# ============================================================
# 4. OUTLIER DETECTION & TREATMENT
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
    Detect and handle outliers. Returns (processed_df, outliers_count, bounds_dict).
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols:
        return result, 0, {}

    method = method.lower()
    action = action.lower()
    total_outliers = 0
    bounds_dict = {}

    if method == "iqr":
        for col in valid_cols:
            q1 = float(result[col].quantile(0.25))
            q3 = float(result[col].quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)
            bounds_dict[col] = (lower_bound, upper_bound)

            outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
            total_outliers += int(outlier_mask.sum())

            if action == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == "remove":
                result = result[~outlier_mask].copy()

    elif method == "zscore":
        for col in valid_cols:
            mean = float(result[col].mean())
            std = float(result[col].std())
            if std == 0 or np.isnan(std):
                continue
            lower_bound = mean - (factor * std)
            upper_bound = mean + (factor * std)
            bounds_dict[col] = (lower_bound, upper_bound)

            z_scores = (result[col] - mean) / std
            outlier_mask = z_scores.abs() > factor
            total_outliers += int(outlier_mask.sum())

            if action == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            elif action == "remove":
                result = result[~outlier_mask].copy()

    elif method == "winsorize":
        for col in valid_cols:
            lower_bound = float(result[col].quantile(lower_pct))
            upper_bound = float(result[col].quantile(1.0 - upper_pct))
            bounds_dict[col] = (lower_bound, upper_bound)

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

    return result, total_outliers, bounds_dict


# ============================================================
# 5. CATEGORICAL ENCODING
# ============================================================

def apply_categorical_encoding(df: pd.DataFrame, decisions: dict) -> tuple:
    """
    Encode categorical columns based on strategy mapping.
    Returns: (encoded_df, encoders_dict)
    """
    result = df.copy()
    encoders_dict = {}

    for col, method in decisions.items():
        if col not in result.columns:
            continue

        method = str(method).lower()

        if method in ["one_hot", "one-hot", "ohe"]:
            # Capture unique non-null categories
            unique_cats = [str(x) for x in result[col].dropna().unique()]
            if len(unique_cats) > 30:
                top_cats = list(result[col].value_counts().nlargest(29).index)
                result[col] = result[col].apply(lambda x: str(x) if x in top_cats else "Other")
                unique_cats = [str(x) for x in top_cats] + ["Other"]

            dummies = pd.get_dummies(result[col].astype(str), prefix=col, drop_first=False, dtype=int)
            dummy_cols = list(dummies.columns)
            dummy_to_cat = {d_col: cat for d_col, cat in zip(dummy_cols, unique_cats)}

            encoders_dict[col] = {
                "type": "one_hot",
                "categories": unique_cats,
                "dummy_cols": dummy_cols,
                "dummy_to_cat": dummy_to_cat,
            }

            result = pd.concat([result.drop(columns=[col]), dummies], axis=1)

        elif method == "label":
            le = LabelEncoder()
            filled_col = result[col].fillna("Missing").astype(str)
            result[col] = le.fit_transform(filled_col)
            mapping = {str(cls_): int(idx) for idx, cls_ in enumerate(le.classes_)}
            encoders_dict[col] = {
                "type": "label",
                "obj": le,
                "categories": list(le.classes_),
                "mapping": mapping,
            }

        elif method == "ordinal":
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            filled_col = result[[col]].fillna("Missing").astype(str)
            result[col] = oe.fit_transform(filled_col).ravel()
            categories = list(oe.categories_[0])
            mapping = {str(cat): float(i) for i, cat in enumerate(categories)}
            encoders_dict[col] = {
                "type": "ordinal",
                "obj": oe,
                "categories": categories,
                "mapping": mapping,
            }

        elif method == "frequency":
            freq_map = result[col].value_counts(normalize=True).to_dict()
            result[col] = result[col].map(freq_map).fillna(0.0)
            encoders_dict[col] = {
                "type": "frequency",
                "mapping": freq_map,
            }

    return result, encoders_dict


# ============================================================
# 6. FEATURE TRANSFORMATIONS & POLYNOMIAL
# ============================================================

def transform_features(df: pd.DataFrame, columns: list, method: str = "log1p") -> tuple:
    """
    Apply mathematical transformations: 'log1p', 'box-cox', 'yeo-johnson'.
    Returns: (transformed_df, transforms_dict)
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols:
        return result, {}

    method = method.lower()
    transforms_dict = {}

    if method in ["log", "log1p"]:
        for col in valid_cols:
            min_val = result[col].min()
            shift = (abs(min_val) + 1.0) if (pd.notna(min_val) and min_val <= 0) else 0.0
            if shift > 0:
                result[col] = np.log1p(result[col] + shift)
            else:
                result[col] = np.log1p(result[col])
            transforms_dict[col] = {"method": "log1p", "shift": shift, "transformer": None}

    elif method == "box-cox":
        for col in valid_cols:
            values = result[col].to_numpy(dtype=float)
            min_val = np.nanmin(values)
            shift = (abs(min_val) + 1.0) if (pd.notna(min_val) and min_val <= 0) else 0.0
            if shift > 0:
                values = values + shift
            pt = PowerTransformer(method="box-cox", standardize=True)
            result[col] = pt.fit_transform(values.reshape(-1, 1)).ravel()
            transforms_dict[col] = {"method": "box-cox", "shift": shift, "transformer": pt}

    elif method == "yeo-johnson":
        for col in valid_cols:
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            result[[col]] = pt.fit_transform(result[[col]])
            transforms_dict[col] = {"method": "yeo-johnson", "shift": 0.0, "transformer": pt}

    return result, transforms_dict


def generate_polynomial_features(
    df: pd.DataFrame,
    columns: list,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False
) -> tuple:
    """Generate polynomial and interaction terms. Returns (result_df, poly_transformer)."""
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols or degree < 1:
        return result, None

    if len(valid_cols) > 15 and degree > 2:
        raise ValueError("Too many columns selected for degree > 2 expansion.")

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    poly_data = poly.fit_transform(result[valid_cols])
    poly_feature_names = poly.get_feature_names_out(valid_cols)

    poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=result.index)
    result = pd.concat([result.drop(columns=valid_cols), poly_df], axis=1)
    return result, poly


# ============================================================
# 7. DIMENSIONALITY REDUCTION (PCA) & RFE
# ============================================================

def apply_pca(df: pd.DataFrame, columns: list, n_components=0.95, standardize: bool = True) -> tuple:
    """
    Apply PCA to numerical features.
    Returns: (pca_df, pca_model, scaler_model, explained_variance_ratio)
    """
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        raise ValueError("PCA requires numerical features.")

    X = df[valid_cols].copy()
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_mat = scaler.fit_transform(X)
    else:
        X_mat = X.to_numpy()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_mat)
    pc_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pc_names, index=df.index)

    other_cols = [c for c in df.columns if c not in valid_cols]
    if other_cols:
        pca_df = pd.concat([df[other_cols], pca_df], axis=1)

    return pca_df, pca, scaler, pca.explained_variance_ratio_


def apply_rfe_selection(X: pd.DataFrame, y: pd.Series, n_features: int = 5, problem_type: str = "Classification") -> tuple:
    """Apply Recursive Feature Elimination (RFE)."""
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
# 8. CLASS IMBALANCE HANDLING
# ============================================================

def handle_imbalanced_classes(X_train: pd.DataFrame, y_train: pd.Series, method: str = "smote") -> tuple:
    """Balance classification dataset using SMOTE or Random Under-sampling."""
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
# 9. FEATURE SCALING
# ============================================================

def apply_feature_scaling(df: pd.DataFrame, columns: list, scaler_type: str = "standard") -> tuple:
    """
    Scale numeric features using Standard, MinMax, Robust, MaxAbs, or Normalizer.
    Returns: (scaled_df, scaler_object)
    """
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not valid_cols:
        return result, None

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
    return result, scaler
