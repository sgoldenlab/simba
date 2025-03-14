__author__ = "Simon Nilsson"


from enum import Enum

import numpy as np


class Unsupervised(Enum):
    ALL_FEATURES_EX_POSE = "ALL FEATURES (EXCLUDING POSE)"
    DATA_SLICE_SELECTION = "data_slice"
    CLF_SLICE_SELECTION = "clf_slice"
    ALL_FEATURES_EXCLUDING_POSE = "ALL FEATURES (EXCLUDING POSE)"
    ALL_FEATURES_INCLUDING_POSE = "ALL FEATURES (INCLUDING POSE)"
    USER_DEFINED_SET = "USER-DEFINED FEATURE SET"
    NAMES = "NAMES"
    START_FRAME = "START_FRAME"
    END_FRAME = "END_FRAME"
    CLASSIFIER = "CLASSIFIER"
    PROBABILITY = "PROBABILITY"
    FRAME = "FRAME"
    VIDEO = "VIDEO"
    FEATURE_PATH = "feature_path"
    BOUT_AGGREGATION_TYPE = "bout_aggregation_type"
    MIN_BOUT_LENGTH = "min_bout_length"
    N_NEIGHBORS = "n_neighbors"
    HASHED_NAME = "HASH"
    DATA = "DATA"
    RAW = "RAW"
    UMAP = "UMAP"
    HDBSCAN = "HDBSCAN"
    TSNE = "TSNE"
    SCALER_TYPE = "SCALER_TYPE"
    CSV = "CSV"
    MULTICOLLINEARITY = "multicollinearity"
    FORMAT = "format"
    SCALED_DATA = "SCALED_DATA"
    PARAMETERS = "PARAMETERS"
    METHODS = "METHODS"
    DR_MODEL = "DR_MODEL"
    MODEL = "MODEL"
    MIN_DISTANCE = "min_distance"

    EUCLIDEAN = "euclidean"
    FEATURE_NAMES = "FEATURE_NAMES"
    SPREAD = "spread"
    SCALER = "scaler"
    SCALED = "scaled"
    VARIANCE = "variance"
    HYPERPARAMETERS = [N_NEIGHBORS, MIN_DISTANCE, SPREAD, SCALER, VARIANCE]
    FRAME_FEATURES = "FRAME_FEATURES"
    FRAME_POSE = "FRAME_POSE"
    FRAME_TARGETS = "FRAME_TARGETS"
    BOUTS_FEATURES = "BOUTS_FEATURES"
    BOUTS_TARGETS = "BOUTS_TARGETS"
    DATASET_DATA_FIELDS = [
        FRAME_FEATURES,
        FRAME_POSE,
        FRAME_TARGETS,
        BOUTS_FEATURES,
        BOUTS_TARGETS,
    ]
    MIN_MAX = "MIN-MAX"
    STANDARD = "STANDARD"
    QUANTILE = "QUANTILE"
    LOW_VARIANCE_FIELDS = "LOW_VARIANCE_FIELDS"


class Clustering(Enum):
    ALPHA = "alpha"
    MIN_CLUSTER_SIZE = "min_cluster_size"
    MIN_SAMPLES = "min_samples"
    EPSILON = "cluster_selection_epsilon"
    CLUSTER_MODEL = "CLUSTER_MODEL"


class UMLOptions(Enum):
    FEATURE_SLICE_OPTIONS = [
        Unsupervised.ALL_FEATURES_INCLUDING_POSE.value,
        Unsupervised.ALL_FEATURES_EXCLUDING_POSE.value,
        Unsupervised.USER_DEFINED_SET.value,
    ]
    BOUT_AGGREGATION_METHODS = ["MEAN", "MEDIAN"]
    UMAP_ADDITIONAL_DATA = [
        "NONE",
        "SCALED FEATURES",
        "RAW FEATURES",
        "SCALED FEATURES & CLASSIFICATIONS",
        "RAW FEATURES & CLASSIFICATIONS",
    ]
    DATA_FORMATS = ["NONE", "SCALED", "RAW"]
    SAVE_FORMATS = ["CSV", "PICKLE"]
    CORRELATION_OPTIONS = ["SPEARMAN", "PEARSONS", "KENDALL"]
    SCALER_OPTIONS = ["MIN-MAX", "STANDARD", "QUANTILE"]
    CATEGORICAL_OPTIONS = ["VIDEO", "CLASSIFIER", "CLUSTER"]
    CONTINUOUS_OPTIONS = ["START_FRAME", "END_FRAME", "PROBABILITY"]
    SPEED_OPTIONS = [round(x, 1) for x in list(np.arange(0.1, 2.1, 0.1))]
    SHAP_CLUSTER_METHODS = ["Paired clusters", "One-against-all"]
    DR_ALGO_OPTIONS = ["UMAP", "TSNE"]
    CLUSTERING_ALGO_OPTIONS = ["HDBSCAN"]
    VARIANCE_OPTIONS = list(range(0, 100, 10))
    GRAPH_CNT = list(range(1, 11))
    SCATTER_SIZE = list(range(10, 110, 10))
    SHAP_SAMPLE_OPTIONS = list(range(100, 1100, 100))
    DATA_TYPES = [
        "BOUTS FEATURES",
        "BOUTS TARGETS",
        "BOUTS DIMENSIONALITY REDUCTION DATA",
        "BOUTS CLUSTER LABELS",
        "CLUSTERER HYPER-PARAMETERS",
        "DIMENSIONALITY REDUCTION HYPER-PARAMETERS",
        "FEATURE NAMES",
        "FRAME FEATURES",
        "FRAME POSE",
        "FRAME TARGETS",
        "LOW VARIANCE FIELDS",
        "SCALER",
        "SCALED DATA",
    ]
