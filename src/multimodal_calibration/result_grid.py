"""Strict expected-grid validation shared by paper result aggregators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

try:
    from .experiment_config import (
        AUXILIARY_PAPER_RUN_CONFIG,
        DATASET_MODALITY_FIELDS,
        MISSING_PAPER_RUN_CONFIG,
        PAPER_DATASETS,
        PAPER_NEURAL_ARCHITECTURES,
        PAPER_SEEDS,
        TESTSPLIT_PAPER_RUN_CONFIG,
        TESTSPLIT_RESULT_METHODS,
        canonical_run_config,
    )
except ImportError:  # Direct-script imports used by experiment drivers.
    from experiment_config import (  # type: ignore[no-redef]
        AUXILIARY_PAPER_RUN_CONFIG,
        DATASET_MODALITY_FIELDS,
        MISSING_PAPER_RUN_CONFIG,
        PAPER_DATASETS,
        PAPER_NEURAL_ARCHITECTURES,
        PAPER_SEEDS,
        TESTSPLIT_PAPER_RUN_CONFIG,
        TESTSPLIT_RESULT_METHODS,
        canonical_run_config,
    )
HETERO_CORE_METHODS = {
    "solo_tab",
    "homo_xgb_concat",
    "homog_xgb_gated",
    "hetero_linear_stacker",
    "hetero_gated_mixture",
}
HETERO_EXPECTED_METHODS = {
    dataset: HETERO_CORE_METHODS
    | {
        f"solo_{modality}"
        for modality in ("text", "image")
        if DATASET_MODALITY_FIELDS[dataset][modality]
    }
    for dataset in PAPER_DATASETS
}
AUXILIARY_NN_METHODS = frozenset(PAPER_NEURAL_ARCHITECTURES)
AUXILIARY_CAMPAIGN_METHODS = frozenset(
    set().union(*HETERO_EXPECTED_METHODS.values()) | AUXILIARY_NN_METHODS
)
MISSING_EXPECTED_METHODS = tuple(MISSING_PAPER_RUN_CONFIG["methods"])


def _missing_expected_regime_missing() -> dict[str, dict[str, tuple[str, ...]]]:
    """Derive the paper's missingness regimes from each dataset's modalities."""
    expected: dict[str, dict[str, tuple[str, ...]]] = {}
    for dataset in PAPER_DATASETS:
        fields = DATASET_MODALITY_FIELDS[dataset]
        present = tuple(
            modality for modality in ("image", "text") if fields[modality]
        )
        regimes: dict[str, tuple[str, ...]] = {"full": ()}
        for modality in ("text", "image"):
            if fields[modality]:
                regimes[f"no_{modality}"] = (modality,)
        if present:
            regimes["tab_only"] = present
        expected[dataset] = regimes
    return expected


MISSING_EXPECTED_REGIME_MISSING = _missing_expected_regime_missing()


def testsplit_expected_grid(
    seeds: Iterable[int] = PAPER_SEEDS,
) -> set[tuple[int, str, str, str, str]]:
    """Return the canonical held-out test-split output cells."""
    config = TESTSPLIT_PAPER_RUN_CONFIG
    semf_strata = {
        "regime": tuple(config["semf_regimes"]),
        "uncert": tuple(
            str(index) for index in range(config["uncertainty_bins"])
        ),
        "region": tuple(
            str(index) for index in range(config["region_clusters"])
        ),
    }
    augmented_strata = {
        "regime": ("full",),
        "uncert": semf_strata["uncert"],
    }
    strata_by_family = {
        "semf_testsplit": semf_strata,
        "aug_theta_testsplit": augmented_strata,
    }
    cells = {
        (int(seed), family, analysis, method, stratum)
        for seed in seeds
        for family, analyses in TESTSPLIT_RESULT_METHODS.items()
        for analysis, methods in analyses.items()
        for method in methods
        for stratum in strata_by_family[family][analysis]
    }
    if config["include_density"]:
        cells.update(
            (int(seed), family, "regime", "density", stratum)
            for seed in seeds
            for family, strata in (
                ("semf_testsplit", semf_strata["regime"]),
                ("aug_theta_testsplit", augmented_strata["regime"]),
            )
            for stratum in strata
        )
    return cells


def auxiliary_run_config(
    alpha: float,
    tune_frac: float,
    calib_frac: float | None = None,
) -> dict:
    """Full shared configuration for heterogeneous and neural workers."""
    config = canonical_run_config("aux")
    config["alpha"] = float(alpha)
    if calib_frac is not None:
        config["split"]["calib_frac"] = float(calib_frac)
    config["split"]["tune_frac"] = float(tune_frac)
    return config


def expected_seeds(
    seeds: Iterable[int] | None = None,
    *,
    expected_seed_count: int = len(PAPER_SEEDS),
) -> tuple[int, ...]:
    values = tuple(sorted(set(PAPER_SEEDS if seeds is None else map(int, seeds))))
    if len(values) != expected_seed_count:
        raise ValueError(
            f"expected exactly {expected_seed_count} distinct seeds, got "
            f"{len(values)}: {list(values)}"
        )
    return values


def paper_campaign_scope(
    execution_datasets: Iterable[str],
    execution_seeds: Iterable[int],
    *,
    campaign_datasets: Iterable[str] | None = None,
    campaign_seeds: Iterable[int] | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Resolve the full configured dataset-by-seed paper campaign scope."""
    run_datasets = tuple(sorted(set(execution_datasets)))
    run_seeds = tuple(sorted(set(map(int, execution_seeds))))
    scope_datasets = tuple(sorted(set(
        run_datasets if campaign_datasets is None else campaign_datasets
    )))
    scope_seeds = expected_seeds(
        run_seeds if campaign_seeds is None else campaign_seeds
    )
    if set(scope_datasets) != set(PAPER_DATASETS):
        raise ValueError(
            "auxiliary paper campaign must declare exactly datasets "
            f"{list(PAPER_DATASETS)}, got {list(scope_datasets)}"
        )
    if tuple(scope_seeds) != PAPER_SEEDS:
        raise ValueError(
            "paper campaign must declare exactly seeds "
            f"{list(PAPER_SEEDS)}, got {list(scope_seeds)}"
        )
    if not set(run_datasets).issubset(scope_datasets):
        raise ValueError("execution datasets are outside the declared campaign")
    if not set(run_seeds).issubset(scope_seeds):
        raise ValueError("execution seeds are outside the declared campaign")
    return scope_datasets, scope_seeds


# Backward-compatible descriptive alias for the shared auxiliary workflow.
auxiliary_campaign_scope = paper_campaign_scope


def require_dataset_seed_grid(
    payloads: Mapping[str, Mapping[int, object]],
    datasets: Iterable[str],
    seeds: Iterable[int],
    *,
    label: str,
) -> None:
    """Raise with a complete diagnostic when dataset×seed cells are absent."""
    dataset_set = tuple(sorted(set(datasets)))
    seed_set = tuple(sorted(set(map(int, seeds))))
    missing = [
        f"{dataset}/seed{seed}"
        for dataset in dataset_set
        for seed in seed_set
        if seed not in payloads.get(dataset, {})
    ]
    extras = [
        f"{dataset}/seed{seed}"
        for dataset, seedmap in payloads.items()
        for seed in seedmap
        if dataset not in dataset_set or seed not in seed_set
    ]
    if missing or extras:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if extras:
            parts.append("unexpected: " + ", ".join(extras))
        raise ValueError(f"incomplete {label} grid ({'; '.join(parts)})")


def require_methods(payload: dict, methods: Iterable[str], *, label: str) -> None:
    results = payload.get("results", {})
    missing = sorted(set(methods) - set(results))
    if missing:
        raise ValueError(f"{label} missing methods: {', '.join(missing)}")


def require_expected_method_grid(
    payloads: Mapping[str, Mapping[int, dict]],
    expected_by_dataset: Mapping[str, Iterable[str]],
    seeds: Iterable[int],
    *,
    label: str,
) -> None:
    """Require an explicit method set in every dataset×seed cell."""
    for dataset, methods in expected_by_dataset.items():
        expected = set(methods)
        for seed in seeds:
            actual = set(
                payloads.get(dataset, {}).get(int(seed), {}).get("results", {})
            )
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"{label} {dataset}/seed{seed} method grid differs; "
                    f"missing={missing}, unexpected={extra}"
                )
