"""Canonical configuration for the paper experiments.

This module is intentionally dependency-light.  Experiment drivers, the
reproduction pipeline, and the offline verifier all import the same values so
that command-line defaults cannot silently drift from the reported setup.
"""

from __future__ import annotations

from copy import deepcopy
import json
from types import MappingProxyType
from typing import Mapping


# Identifier of the completed frozen reproduction. New reproductions must use
# their own explicit tag through reproduce_paper.py.
FROZEN_CAMPAIGN_TAG = "camera_ready_final_20260806"
PAPER_DATASETS = ("sred", "mercari", "pawpularity", "imdb_wiki")
PAPER_SEEDS = (0, 1, 2, 3, 4)
PAPER_NEURAL_ARCHITECTURES = ("mlp_concat", "mm_attn_nn")
PAPER_CPU_DEVICE = "cpu"
PAPER_DISAGREEMENT_BINS = 3
PAPER_XGB_TREE_METHOD = "hist"
PAPER_XGB_EVAL_METRIC = "rmse"
PAPER_XGB_POINT_OBJECTIVE = "reg:squarederror"
SRED_TABULAR_COLUMNS = ("living_space", "rooms", "lat", "lon")
SRED_REGION_COLUMNS = ("lat", "lon")
PREDAGN_BASE_PREDICTORS = (
    "xgb_point",
    "xgb_quantile",
    "sourcewise_stack",
    "sourcewise_quantile",
)
PREDAGN_HEADLINE_BASES = (
    "xgb_point",
    "xgb_quantile",
    "sourcewise_quantile",
)
PREDAGN_SEED_POLICY: Mapping[str, int] = MappingProxyType({
    "modality_point_stride": 17,
    "modality_quantile_stride": 101,
    "difficulty_xgb_point": 10_000,
    "difficulty_xgb_quantile": 20_000,
    "difficulty_sourcewise_stack": 30_000,
    "difficulty_sourcewise_quantile": 40_000,
})
PREDAGN_IMPLEMENTATION_POLICY = {
    "categorical_top_k": 50,
    "robust_scale_quantiles": (0.25, 0.75),
    "binned_scale_ratios": (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0),
}


def campaign_results_relative(campaign_tag: str) -> str:
    """Return the result directory while preserving the frozen run's identity."""
    namespace = (
        "camera_ready"
        if campaign_tag == FROZEN_CAMPAIGN_TAG
        else "reproduction"
    )
    return f"results/{namespace}/{campaign_tag}"


def _point_xgb_config() -> dict:
    return {
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "early_stopping_rounds": 50,
        "eval_metric": PAPER_XGB_EVAL_METRIC,
        "objective": PAPER_XGB_POINT_OBJECTIVE,
        "tree_method": PAPER_XGB_TREE_METHOD,
    }


PREDAGN_PAPER_RUN_CONFIG = {
    "experiment": "predictor_agnostic_v1",
    "alpha": 0.05,
    "split": {"calib_frac": 0.20, "tune_frac": 0.15, "min_fold": 50},
    "point_xgb": _point_xgb_config(),
    "quantile_xgb": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:quantileerror",
        "quantiles": [0.025, 0.5, 0.975],
        "tree_method": PAPER_XGB_TREE_METHOD,
    },
    "sourcewise_ridge_alpha": 1.0,
    "difficulty_model": {
        "model": "xgboost_log_residual_scale",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": PAPER_XGB_TREE_METHOD,
        "device": PAPER_CPU_DEVICE,
        "n_jobs": 1,
        "target_floor": 1e-6,
    },
    "weighted_scale": {
        "parameterization": "a0=1; gamma=a1/a0",
        "a0_source_grid": [0.001, 0.01, 0.1, 0.5, 1.0, 3.0],
        "a1_source_grid": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        "score": "pure_expansion",
    },
    "calibrations": [
        "marginal",
        "mondrian_dis",
        "weighted_dis",
        "difficulty_weighted",
        "binned_weighted_dis",
    ],
}

MISSING_PAPER_RUN_CONFIG = {
    "experiment": "missing_modality_v1",
    "alpha": 0.05,
    "split": {"calib_frac": 0.20, "tune_frac": 0.15, "min_fold": 50},
    "point_xgb": _point_xgb_config(),
    "methods": [
        "global_full_cal",
        "pooled_masked_cal",
        "mask_matched_cal",
    ],
    "mask_value_after_standardization": 0.0,
}

AUXILIARY_PAPER_RUN_CONFIG = {
    "experiment": "auxiliary_benchmarks_v1",
    "alpha": 0.05,
    "split": {"calib_frac": 0.20, "tune_frac": 0.15, "min_fold": 50},
    "heterogeneous": {
        "xgb_quantile": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators_by_train_size": {
                "tiny_lt_1000": 200,
                "standard_le_50000": 400,
                "large_gt_50000": 300,
            },
        },
        "modality_mlp_quantile": {
            "hidden": 256,
            "dropout": 0.1,
            "epochs": 50,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "gated_mixture": {
            "hidden": 128,
            "learning_rate": 0.001,
            "epochs_standard": 80,
            "epochs_large": 60,
            "epochs_tiny": 200,
        },
        "weighted_a1_grid": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    },
    "neural": {
        "optimizer": "AdamW",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "weight_decay": 0.0001,
        "dropout": 0.2,
        "internal_validation_fraction": 0.10,
        "mlp_concat": {
            "hidden": [512, 256, 128],
            "learning_rate": 0.001,
        },
        "mm_attn_nn": {
            "d_token": 64,
            "n_heads": 4,
            "n_layers": 2,
            "learning_rate": 0.0005,
        },
    },
}

# Size-dependent branches used by the auxiliary implementations. They live
# here (rather than as literals in the drivers) so a paper run has one source
# of truth even for branches not exposed as command-line flags.
AUXILIARY_XGB_CONTROL_CONFIG = {
    "large_train_threshold": 30_000,
    "tiny_train_threshold": 500,
    "n_estimators_large": 200,
    "n_estimators_standard": 300,
    "n_estimators_tiny": 150,
}
AUXILIARY_SIZE_POLICY = {
    "xgb_tiny_upper_exclusive": 1_000,
    "xgb_large_lower_exclusive": 50_000,
    "gate_tiny_upper_exclusive": 500,
    "gate_large_lower_exclusive": 10_000,
}
AUXILIARY_GATE_BATCH_POLICY = {
    "maximum": 256,
    "minimum": 32,
    "tune_divisor": 4,
}
AUXILIARY_COMPOSITION_POLICY = {
    "linear_optimizer": "Nelder-Mead",
    "linear_max_iterations": 400,
    "linear_x_tolerance": 0.0001,
    "gate_optimizer": "AdamW",
    "gate_weight_decay": 0.0001,
    "gate_dropout": 0.1,
    "gate_internal_validation_fraction": 0.15,
    "gate_early_stopping_patience": 12,
    "gate_early_stopping_min_delta": 0.00001,
}
AUXILIARY_NEURAL_TINY_POLICY = {
    "train_threshold": 1_000,
    "epochs": 50,
    "patience": 8,
    "maximum_batch_size": 64,
    "dropout": 0.4,
}
AUXILIARY_NEURAL_EVAL_BATCH_SIZE = 512
AUXILIARY_NEURAL_WARMUP_FRACTION = 0.05
AUXILIARY_NEURAL_TRAINING_POLICY = {
    "optimizer": "AdamW",
    "scheduler": "cosine_warmup",
    "early_stopping_min_delta": 0.000001,
    "epoch_shuffle_seed_stride": 7919,
    "attention_feedforward_expansion": 2,
}
AUXILIARY_PREDICTOR_POLICY = {
    "optimizer": "AdamW",
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.9,
    "xgb_tree_method": PAPER_XGB_TREE_METHOD,
    "mlp_internal_validation_fraction": 0.1,
    "mlp_early_stopping_patience": 8,
    "mlp_early_stopping_min_delta": 0.00001,
}
AUXILIARY_PREPROCESSING_POLICY = {
    "categorical_top_k": 50,
}


def auxiliary_xgb_quantile_estimators(train_size: int) -> int:
    """Return the canonical heterogeneous XGB tree budget."""
    budgets = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
        "xgb_quantile"
    ]["n_estimators_by_train_size"]
    if train_size < AUXILIARY_SIZE_POLICY["xgb_tiny_upper_exclusive"]:
        return int(budgets["tiny_lt_1000"])
    if train_size > AUXILIARY_SIZE_POLICY["xgb_large_lower_exclusive"]:
        return int(budgets["large_gt_50000"])
    return int(budgets["standard_le_50000"])


def auxiliary_gate_epochs(train_size: int) -> int:
    """Return the canonical gated-mixture epoch budget."""
    config = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"]["gated_mixture"]
    if train_size < AUXILIARY_SIZE_POLICY["gate_tiny_upper_exclusive"]:
        return int(config["epochs_tiny"])
    if train_size > AUXILIARY_SIZE_POLICY["gate_large_lower_exclusive"]:
        return int(config["epochs_large"])
    return int(config["epochs_standard"])


def auxiliary_xgb_control_estimators(train_size: int) -> int:
    """Return the canonical homogeneous-control XGB tree budget."""
    config = AUXILIARY_XGB_CONTROL_CONFIG
    if train_size < config["tiny_train_threshold"]:
        return int(config["n_estimators_tiny"])
    if train_size > config["large_train_threshold"]:
        return int(config["n_estimators_large"])
    return int(config["n_estimators_standard"])

SEMF_PAPER_RUN_CONFIG = {
    "experiment": "sred_semf_v1",
    "alpha": 0.05,
    "features": {
        "text_pca": 8,
        "image_pca": 8,
        "text_slug": "paraphrase-multilingual-MiniLM-L12-v2",
        "image_slug": "dinov2-vits14",
        "image_types": ["montage_organized", "satellite"],
    },
    "semf": {
        "R": 10,
        "R_infer": 50,
        "max_it": 15,
        "nodes_per_feature": 2,
        "x_group_size": 4,
        "z_norm_sd": "0.1",
        "model_class": "MultiXGBs",
        "weight_alignment": "aligned",
        "outer_patience": 5,
        "allow_partial_recovery": False,
    },
    "tree": {
        "n_estimators": 100,
        "max_depth": None,
        "patience": 0,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "regimes": ["full", "no_text", "no_img", "tab_only"],
    "data_valid_fraction": 0.15,
    "models_valid_fraction": 0.15,
}

AUGMENTED_PAPER_RUN_CONFIG = {
    "experiment": "sred_augmented_theta_v1",
    "R_infer": 50,
    "alpha": 0.05,
    "n_jobs": 4,
    "variants": ["aug_full", "aug_tab", "aug_z"],
    "xgb": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "eval_metric": PAPER_XGB_EVAL_METRIC,
        "tree_method": PAPER_XGB_TREE_METHOD,
    },
    "calibrations": ["raw", "cqr", "density"],
}
AUGMENTED_IMPLEMENTATION_POLICY = {
    "latent_sampling_seed_offset": 1,
}

TESTSPLIT_PAPER_RUN_CONFIG = {
    "experiment": "sred_heldout_testsplit_v1",
    "alpha": 0.05,
    "final_calibration_fraction": 0.5,
    "split_seed_offset": 210_517,
    "include_density": False,
    "uncertainty_bins": 3,
    "region_clusters": 4,
    "kmeans_n_init": 10,
    "semf_regimes": ["full", "no_img", "no_text", "tab_only"],
}
# Output identifiers are implementation contracts rather than tunable
# hyperparameters, so keep them outside the frozen run-config payload while
# still giving every writer, verifier, table, and figure one source of truth.
TESTSPLIT_RESULT_METHODS = MappingProxyType({
    "semf_testsplit": MappingProxyType({
        "regime": ("raw", "global_cqr", "mask_matched_cqr"),
        "uncert": ("global_cqr", "mondrian_cqr"),
        "region": ("global_cqr", "mondrian_cqr"),
    }),
    "aug_theta_testsplit": MappingProxyType({
        "regime": ("raw", "cqr"),
        "uncert": ("global_cqr", "mondrian_cqr"),
    }),
})
# Frozen presentation order for the public masked-regime figure. This is
# intentionally distinct from the computation order in ``semf_regimes``.
TESTSPLIT_FIGURE_REGIME_ORDER = (
    "full",
    "no_text",
    "no_img",
    "tab_only",
)

PREDMOND_PAPER_RUN_CONFIG = {
    "experiment": "sred_predictor_agnostic_mondrian_v1",
    "alpha": 0.05,
    "validation_tune_fraction": 0.5,
    "split_seed_offset": 10_003,
    "xgb_point": {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
    },
    "xgb_quantile": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "modality_solo_xgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
    },
    "weighted_scale_grid": {
        "a0": [0.001, 0.01, 0.1, 0.5, 1.0, 3.0],
        "a1": [0.0, 0.5, 1.0, 3.0, 10.0, 30.0],
    },
    "bases": ["xgb_point", "xgb_quantile", "aug_theta"],
    "calibrations": ["marginal", "mondrian_dis", "weighted_dis"],
}
PREDMOND_IMPLEMENTATION_POLICY = {
    "text_fields": ("header", "ad_description"),
    "minimum_tune_or_calibration_size": 10,
    "modality_solo_validation_fraction": 0.1,
}

WORKED_IMPLEMENTATION_POLICY = {
    "base": "xgb_quantile",
    "calibrations": ("marginal", "weighted_dis"),
    "selection_uses_test_labels": True,
    "example_count": 2,
}
WORKED_PAPER_RUN_CONFIG = {
    "experiment": "worked_examples_v1",
    "dataset": "sred",
    "seed": 0,
    "selection_uses_test_labels": WORKED_IMPLEMENTATION_POLICY[
        "selection_uses_test_labels"
    ],
    "parent_predagn": deepcopy(PREDAGN_PAPER_RUN_CONFIG),
    "base": WORKED_IMPLEMENTATION_POLICY["base"],
    "calibrations": list(WORKED_IMPLEMENTATION_POLICY["calibrations"]),
    "device": PAPER_CPU_DEVICE,
}


def alpha_result_tag(alpha: float) -> str:
    """Return the repository's canonical alpha filename tag."""
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {value}")
    percent = value * 100
    if abs(percent - round(percent)) < 1e-8:
        return f"a{int(round(percent)):02d}"
    suffix = (
        f"{value:.4f}".rstrip("0").rstrip(".").replace("0.", "")
    )
    return "a" + suffix.replace(".", "p")


PREDAGN_RESULT_TAG = alpha_result_tag(PREDAGN_PAPER_RUN_CONFIG["alpha"])
MISSING_RESULT_TAG = alpha_result_tag(MISSING_PAPER_RUN_CONFIG["alpha"])
# Backward-compatible name for scripts that only consume predictor-agnostic
# artifacts. Cross-family code must use the family-specific tags above.
PAPER_RESULT_TAG = PREDAGN_RESULT_TAG
WORKED_EXAMPLE_COUNT = WORKED_IMPLEMENTATION_POLICY["example_count"]

DATASET_LOADER_POLICY = {
    "mercari": {
        "maximum_labeled_rows": 50_000,
        "test_fraction": 0.20,
        "split_seed": 0,
    },
    "pawpularity": {
        "test_fraction": 0.20,
        "split_seed": 0,
    },
    "imdb_wiki": {
        "minimum_face_score": 1.0,
        "minimum_age": 1,
        "maximum_age": 100,
        "hash_bucket_count": 10,
        "train_bucket_count": 9,
    },
}
DATASET_MODALITY_FIELDS = {
    "sred": {
        "text": ("header", "ad_description"),
        "image": ("montage_organized", "satellite"),
    },
    "mercari": {
        "text": ("name", "item_description"),
        "image": (),
    },
    "imdb_wiki": {
        "text": ("name",),
        "image": ("face",),
    },
    "pawpularity": {
        "text": (),
        "image": ("photo",),
    },
}

SEMF_IMPLEMENTATION_POLICY = {
    "preprocessing_train_fraction": 1.0,
    "scale_output": True,
    "return_mean_default": True,
    "stopping_metric": "RMSE",
    "extra_trees_max_depth": 10,
    "mlp": {
        "batch_size": 256,
        "epochs": 200,
        "learning_rate": 0.001,
        "patience": 10,
    },
}

_CANONICAL_RUN_CONFIGS = {
    "predagn": PREDAGN_PAPER_RUN_CONFIG,
    "missing": MISSING_PAPER_RUN_CONFIG,
    "aux": AUXILIARY_PAPER_RUN_CONFIG,
    "semf": SEMF_PAPER_RUN_CONFIG,
    "augmented": AUGMENTED_PAPER_RUN_CONFIG,
    "testsplit": TESTSPLIT_PAPER_RUN_CONFIG,
    "predictor_mondrian": PREDMOND_PAPER_RUN_CONFIG,
    "worked": WORKED_PAPER_RUN_CONFIG,
}
CANONICAL_RUN_CONFIGS: Mapping[str, dict] = MappingProxyType(
    _CANONICAL_RUN_CONFIGS
)
_CANONICAL_SNAPSHOTS = MappingProxyType({
    family: json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for family, config in _CANONICAL_RUN_CONFIGS.items()
})


def canonical_run_config(family: str) -> dict:
    """Return an isolated copy of the immutable paper configuration."""
    try:
        return json.loads(_CANONICAL_SNAPSHOTS[family])
    except KeyError as error:
        raise ValueError(f"unknown paper experiment family: {family}") from error


def require_canonical_run_config(family: str, actual: Mapping) -> None:
    """Fail closed when a paper-producing command uses different settings."""
    expected = canonical_run_config(family)
    if dict(actual) != expected:
        raise ValueError(
            f"{family} hyperparameters differ from the canonical paper "
            "configuration; use paper/scripts/reproduce_paper.py"
        )


# These are the exact 31 downstream inputs used by the audited reference run.
# Stage 0 cache archives are separately auditable, but paper result jobs consume
# only this reviewed allowlist.
PAPER_INPUT_SHA256: Mapping[str, str] = MappingProxyType({
    "data/imdb_wiki/embeddings/test_face_dinov2-vits14.npy":
        "91e3753246fdc893a0f5731502adf29ada7fa27f45bfc798048f19225e285fbc",
    "data/imdb_wiki/embeddings/test_name_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "468dbc24e7ce386de79831ef0f9c2a932f5c0409f7c0058b412829631d6d72d3",
    "data/imdb_wiki/embeddings/train_face_dinov2-vits14.npy":
        "eb83e30c0f863a20f81ba5abcbb14d68e754fe8217d67fbea9f3826e5aac2b35",
    "data/imdb_wiki/embeddings/train_name_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "3cd2a0927c5ae78f94516b3a3929b77e57a2264c36ce2329df47224082c67983",
    "data/imdb_wiki/processed.parquet":
        "99f513eada1f395ebb4949228f57abfab1e24ef8624af84b5dfd45ac80ce1720",
    "data/mercari/embeddings/test_item_description_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "ecf0108a5987784cc0af0bc2d1547dc3c1a47777401285e28ce690befed3f674",
    "data/mercari/embeddings/test_name_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "0d288ff7aa187204546d8bd45978c1fc8ef79fe8bcee122e98aaf650cdae5813",
    "data/mercari/embeddings/train_item_description_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "e2e9563c9a2bd5c81ee5c73ddc25bb5f9bda6444c0c7309140aca668b53e20cb",
    "data/mercari/embeddings/train_name_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "c043e05fa9ec9a9c820bf8e377d52a27b068f8ac7dbd00b89d9cbe4e52d82f84",
    "data/mercari/raw/train.tsv":
        "94b24fbecca842140b6276e9da43ce66be3b56b4b991375228802a74050fe471",
    "data/pawpularity/embeddings/test_photo_dinov2-vits14.npy":
        "1fb350b6b011714d347feb32cd1c251ae0f30f20d030a5b9c097e5670f24583f",
    "data/pawpularity/embeddings/train_photo_dinov2-vits14.npy":
        "891f0c2462e640d903028d3d8eb369cea88aad08c5d259d1838b213e16f2b95c",
    "data/pawpularity/raw/train.csv":
        "e4abd43ce55f42388d3bcb9cfcb97168c6a3387aa8dd694da1d3bf1bfac65d0c",
    "data/sred/embeddings/test_ad_description_multilingual-e5-large.npy":
        "3bd3f8f0320624bdc31b60a07d64b66854a29adf4b33fd1a30f4237e03ae9fe9",
    "data/sred/embeddings/test_ad_description_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "4b45b31aa94da85f47564dc18e0dd3863673f1ff07bc1fa5f66798479e9eb3ab",
    "data/sred/embeddings/test_header_multilingual-e5-large.npy":
        "e4e97cd7fda7e00ffd924d760ee990aab7db8e21f0d1aa33c01b71263fa59e81",
    "data/sred/embeddings/test_header_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "2caf52cbda6fbda9c7706077eec9b18230aa174d7925eb4de21dcc46e4efdf23",
    "data/sred/embeddings/test_montage_organized_dinov2-vitb14.npy":
        "ff5ba17aaa602bf193312df3f10e4d833bf487225bace047640a90a9fbe4f78e",
    "data/sred/embeddings/test_montage_organized_dinov2-vits14.npy":
        "dd307c2ae5cd7d85c8857d15acfc827efdb3679fabb91de4a131be10c3d933f7",
    "data/sred/embeddings/test_satellite_dinov2-vitb14.npy":
        "b01e4ff6db6a6528c688b453652980f9b2e5f7a44d804abdbc56c9b53d7acdcd",
    "data/sred/embeddings/test_satellite_dinov2-vits14.npy":
        "0580a90a3ab1d793c06182e0bcc74cc0dea87510e4bfddfaaf1695e018135496",
    "data/sred/embeddings/train_ad_description_multilingual-e5-large.npy":
        "4fbccc7b086b2c9d94f662efd7200687493fe984903b948ebb1e5d15c399ae36",
    "data/sred/embeddings/train_ad_description_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "c8febdfa3f26935705521a2f30864e8038b81cb7fd5b08000267fd7a2a63c2fa",
    "data/sred/embeddings/train_header_multilingual-e5-large.npy":
        "f6770efd140cc63694fa486e000cac21d93e5fc386feda641485ded63c32d521",
    "data/sred/embeddings/train_header_paraphrase-multilingual-MiniLM-L12-v2.npy":
        "bcc050559f7354b51ddbc27ded02b68518a7be2110edcc04d52c910c9baaec69",
    "data/sred/embeddings/train_montage_organized_dinov2-vitb14.npy":
        "4bad2bda8892fd1fd9b77955993f60ff239d9b40721025b3ee43380e1f5ffdbd",
    "data/sred/embeddings/train_montage_organized_dinov2-vits14.npy":
        "df9e1630b663ed85f894af0e931ad6720b32afeb1bd72b2df5882c61d16492fa",
    "data/sred/embeddings/train_satellite_dinov2-vitb14.npy":
        "008a53e8d856a47278e29856f242a46916d43dce843bd75c0c102f81ab1bc919",
    "data/sred/embeddings/train_satellite_dinov2-vits14.npy":
        "ad20a9d0f271ce548dc32f96eaeb03c953bb0413f78248469c5696eb7afdfaf9",
    "data/sred/metadata/test_data_with_text.csv":
        "a7bf3f20d031a8349e9a7ab1eebe2e0114ae27b544a5a9c85a8c39f130f67435",
    "data/sred/metadata/train_data_with_text.csv":
        "09661d800e239f4e85793751628ad0ac89c2bd70145330dd1c8cd454219f3792",
})


# One ledger record is required for every logical submitted family.  Array
# cardinalities are checked separately from the parent job's exit code.
EXPECTED_JOB_TASKS: Mapping[str, int] = MappingProxyType({
    "preflight": 1,
    "predictor_agnostic": len(PAPER_DATASETS) * len(PAPER_SEEDS),
    "missing_modality": len(PAPER_DATASETS) * len(PAPER_SEEDS),
    "heterogeneous": len(PAPER_DATASETS) * len(PAPER_SEEDS),
    "neural": (
        len(PAPER_DATASETS)
        * len(PAPER_NEURAL_ARCHITECTURES)
        * len(PAPER_SEEDS)
    ),
    "worked_examples": 1,
    "semf": len(PAPER_SEEDS),
    "augmented_theta": 1,
    "testsplit": 1,
    "predictor_mondrian": 1,
    "predagn_aggregate": 1,
    "missing_aggregate": 1,
    "auxiliary_aggregate": 1,
})


# Execution policy is centralized alongside the scientific configuration so
# the portable runner and Slurm dispatcher cannot silently select different
# devices or numerical-library thread modes. Scheduler resources (wall time,
# memory, CPU count, partitions, and GPU model) remain cluster policy.
# Values are immutable tuples: (thread mode, required device class).
EXPERIMENT_FAMILY_EXECUTION_POLICY: Mapping[
    str, tuple[str, str]
] = MappingProxyType({
    "preflight": ("isolated", PAPER_CPU_DEVICE),
    "predictor_agnostic": ("allocated", PAPER_CPU_DEVICE),
    "missing_modality": ("allocated", PAPER_CPU_DEVICE),
    "heterogeneous": ("allocated", "cuda"),
    "neural": ("allocated", "cuda"),
    "worked_examples": ("allocated", PAPER_CPU_DEVICE),
    "semf": ("isolated", PAPER_CPU_DEVICE),
    "augmented_theta": ("isolated", PAPER_CPU_DEVICE),
    "testsplit": ("isolated", PAPER_CPU_DEVICE),
    "predictor_mondrian": ("isolated", PAPER_CPU_DEVICE),
    "predagn_aggregate": ("isolated", PAPER_CPU_DEVICE),
    "missing_aggregate": ("isolated", PAPER_CPU_DEVICE),
    "auxiliary_aggregate": ("isolated", PAPER_CPU_DEVICE),
})
