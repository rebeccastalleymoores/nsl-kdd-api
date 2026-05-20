"""
Preprocessing pipeline for NSL-KDD intrusion detection.

Implements the fit/transform pattern:
- fit(): learns parameters from training data (encoding map, scaler stats, columns)
- transform(): applies learned parameters to any data (train, test, or single request)

The fitted object can be saved to disk and reloaded at inference time, ensuring
the API applies identical preprocessing to what the model was trained on.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder


# Column names for raw NSL-KDD data files (43 columns: 41 features + attack_type + difficulty_level)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'attack_type', 'difficulty_level'
]

# Columns dropped because they are near-constant (>99% same value, no signal)
NEAR_CONSTANT_COLS = ['num_outbound_cmds', 'is_host_login', 'land', 'su_attempted']

# Columns one-hot encoded with drop_first=True
ONEHOT_COLS = ['protocol_type', 'flag']


class NSLKDDPreprocessor:
    """
    Preprocessing pipeline for NSL-KDD intrusion detection data.

    Bundles together all the stateful transformations needed to convert
    raw network connection records into model-ready feature vectors:
        - Smoothed target encoding for the 'service' feature
        - One-hot encoding for protocol_type and flag
        - RobustScaler for numerical features
        - Column ordering enforcement (critical for inference)
    """

    def __init__(self, smoothing_m: int = 100):
        # Smoothing parameter for target encoding. Higher = more conservative.
        # Notebook uses m=100; keep as default but allow override.
        self.smoothing_m = smoothing_m

        # State learned during fit() — all initialised to None until then.
        self.service_encoding_map: dict = None    # service name -> smoothed attack rate
        self.global_attack_rate: float = None     # fallback for unseen services
        self.scaler: RobustScaler = None          # fitted scaler for numerical features
        self.numerical_to_scale: list = None      # exact column names the scaler was fit on
        self.feature_columns: list = None         # full column list the model expects, in order

    def fit(self, df: pd.DataFrame) -> 'NSLKDDPreprocessor':
        """
        Learn preprocessing parameters from training data.

        Expects df to contain raw NSL-KDD columns including 'attack_type'
        (used here only to compute the service encoding map).
        """
        df = df.copy()

        # ----- Step 1: compute service encoding map (smoothed target encoding) -----
        # Binary attack indicator: 1 if any attack, 0 if normal
        is_attack = (df['attack_type'] != 'normal').astype(int)
        self.global_attack_rate = is_attack.mean()

        # Per-service statistics: count and raw attack rate
        service_stats = df.groupby('service').agg(
            total_count=('attack_type', 'size'),
            attack_rate=('attack_type', lambda x: (x != 'normal').mean())
        )

        # Apply smoothing: blends category mean with global mean.
        # Rare services pulled toward global rate; common services trusted more.
        smoothed = (
            (service_stats['total_count'] * service_stats['attack_rate']
             + self.smoothing_m * self.global_attack_rate)
            / (service_stats['total_count'] + self.smoothing_m)
        )
        self.service_encoding_map = smoothed.to_dict()

        # ----- Step 2: prepare a feature-only frame to determine columns + scaler -----
        df_features = self._apply_stateless_transforms(df)
        df_features = self._apply_service_encoding(df_features)
        df_features = self._apply_onehot(df_features)

        # ----- Step 3: identify numerical features to scale and fit the scaler -----
        # Exclude: rate features (already 0-1), one-hot features (binary),
        # and service_encoded (already 0-1).
        rate_cols = [c for c in df_features.columns if 'rate' in c.lower()]
        onehot_cols = [c for c in df_features.columns
                       if c.startswith('protocol_type_') or c.startswith('flag_')]
        excluded = set(rate_cols + onehot_cols + ['service_encoded'])

        self.numerical_to_scale = [
            c for c in df_features.select_dtypes(include=['int64', 'float64']).columns
            if c not in excluded
        ]

        self.scaler = RobustScaler()
        self.scaler.fit(df_features[self.numerical_to_scale])

        # ----- Step 4: lock in the final feature column order -----
        # This is what the model will be trained on, and what every inference
        # request will be reindexed to match.
        self.feature_columns = df_features.columns.tolist()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned preprocessing to any data (train, test, or single inference request).

        Returns a DataFrame with columns in the exact order self.feature_columns.
        """
        if self.service_encoding_map is None:
            raise RuntimeError("Preprocessor must be fit before transform.")

        df = df.copy()

        # Apply the same transformations as in fit, in the same order
        df = self._apply_stateless_transforms(df)
        df = self._apply_service_encoding(df)
        df = self._apply_onehot(df)

        # Reindex to match training columns exactly.
        # Missing columns (e.g. a one-hot value that wasn't in this single request)
        # are filled with 0. Extra columns are dropped. Order is enforced.
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Apply scaling using the fitted scaler
        df[self.numerical_to_scale] = self.scaler.transform(df[self.numerical_to_scale])

        return df

    # ----- Internal helpers -----

    def _apply_stateless_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops that don't need any learned state. Safe to apply at any time."""
        # Drop training-only metadata if present (inference requests won't have these)
        drop_if_present = ['attack_type', 'difficulty_level', 'attack_category']
        cols_to_drop = [c for c in drop_if_present if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Drop near-constant features
        df = df.drop(columns=[c for c in NEAR_CONSTANT_COLS if c in df.columns])
        return df

    def _apply_service_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map service name to smoothed attack rate. Unseen services get global rate."""
        df['service_encoded'] = df['service'].map(self.service_encoding_map)
        df['service_encoded'] = df['service_encoded'].fillna(self.global_attack_rate)
        df = df.drop(columns=['service'])
        return df

    def _apply_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode protocol_type and flag with drop_first=True."""
        df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True, dtype=int)
        return df