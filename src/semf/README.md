# SEMF

This directory contains a vendored, project-adapted copy of the MIT-licensed
`semf` implementation accompanying:

> Ilia Azizi, Marc-Olivier Boldi, Valérie Chavez-Demoulin.
> *SEMF: Supervised Expectation-Maximization Framework for Predicting Intervals.*
> Proceedings of the Fourteenth Symposium on Conformal and Probabilistic
> Prediction with Applications, PMLR 266:250-281, 2025.
> <https://proceedings.mlr.press/v266/azizi25a.html>

This copy contains the reproduction-specific adaptations and safeguards used
by this project. In particular, it provides aligned replication weights with
an explicit legacy option, fail-fast training with opt-in partial recovery,
and propagation of the configured XGBoost training options.

Two active experiment drivers under `src/sred/` import it at runtime:

- `run_full_experiment.py` uses `semf.preprocessing.DataPreprocessor` and
  `semf.semf.SEMF` to train SEMF on SRED.
- `augmented_theta.py` uses utilities from `semf.utils`.

See [LICENSE](LICENSE) for the vendored package's MIT license.
