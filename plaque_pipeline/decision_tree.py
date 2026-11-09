from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)

from .config import Config, CFG
from .io_utils import ensure_dir


# =============================================================================
# FIXED HISTORICAL DECISION TREE
# =============================================================================

# These are the fixed parameters used in the historical analysis.
#
# IMPORTANT:
# There is NO GridSearchCV here.
#
# Every model uses exactly these settings.

TREE_PARAMS = {
    "criterion": "entropy",
    "max_depth": 8,
    "min_samples_leaf": 4,
    "min_samples_split": 10,
    "random_state": 0,
}


# Fixed 80/20 split settings.
TEST_SIZE = 0.20
RANDOM_STATE = 0


# =============================================================================
# HELPERS
# =============================================================================


def canonicalise_genotype(
    value: str,
    cfg: Config,
) -> str:

    value = str(value).strip()

    return cfg.genotype_map.get(
        value,
        value,
    )


def _safe_cv_folds(
    y: pd.Series,
    requested: int,
) -> int:

    counts = y.value_counts()

    if counts.empty or counts.min() < 2:

        raise ValueError(
            "Need at least 2 samples in each training class "
            "for cross-validation."
        )

    return max(
        2,
        min(
            requested,
            int(counts.min()),
        ),
    )


def new_fixed_tree() -> DecisionTreeClassifier:
    """
    Return a fresh copy of the fixed historical decision tree.
    """

    return DecisionTreeClassifier(
        **TREE_PARAMS
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def train_decision_tree(
    cfg: Config = CFG,
) -> None:

    ensure_dir(
        cfg.decision_tree_dir
    )

    # -------------------------------------------------------------------------
    # Load morphology feature table
    # -------------------------------------------------------------------------

    if not cfg.feature_table_path.exists():

        raise FileNotFoundError(
            f"Feature table not found: {cfg.feature_table_path}\n"
            "Run `python main.py morphology` first."
        )

    df = pd.read_csv(
    cfg.feature_table_path
    )

    if "image_name" in df.columns:
        df = df[
            ~df["image_name"].isin(
                cfg.excluded_images
            )
        ].copy()

    if "Genotype" not in df.columns:

        raise ValueError(
            "decision_tree_features.csv must contain a Genotype column."
        )

    df = df.copy()

    # -------------------------------------------------------------------------
    # Standardise genotype names
    # -------------------------------------------------------------------------

    df["Genotype"] = df["Genotype"].map(
        lambda x: canonicalise_genotype(
            x,
            cfg,
        )
    )

    # Only train on the requested APP-WT / APP-FIRE classes.
    df = df[
        df["Genotype"].isin(
            cfg.train_labels
        )
    ].copy()

    # -------------------------------------------------------------------------
    # FEATURES
    # -------------------------------------------------------------------------
    #
    # IMPORTANT:
    # The feature selection has NOT been changed.
    #
    # We use exactly whatever is defined in:
    #
    #     cfg.model_features
    #
    # -------------------------------------------------------------------------

    features = list(
        cfg.model_features
    )

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Requested model features are missing from "
            f"the feature table: {missing}"
        )

    model_df = df[
        [
            "image_name",
            "Genotype",
            *features,
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Force features to numeric
    # -------------------------------------------------------------------------

    for column in features:

        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce",
        )

    before = len(
        model_df
    )

    model_df = (
        model_df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=(
                features
                + ["Genotype"]
            )
        )
        .reset_index(
            drop=True
        )
    )

    dropped = (
        before
        - len(model_df)
    )

    model_df.to_csv(
        cfg.decision_tree_dir
        / "model_input_complete_cases.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Check classes
    # -------------------------------------------------------------------------

    if model_df[
        "Genotype"
    ].nunique() != 2:

        raise ValueError(
            "Expected exactly two training classes, found "
            f"{model_df['Genotype'].unique().tolist()}"
        )

    print(
        "\nDecision-tree training data:"
    )

    print(
        model_df[
            "Genotype"
        ].value_counts()
    )

    print(
        f"\nComplete plaques: {len(model_df)}"
    )

    print(
        f"Dropped incomplete plaques: {dropped}"
    )

    # -------------------------------------------------------------------------
    # X / y
    # -------------------------------------------------------------------------

    X = model_df[
        features
    ]

    y = model_df[
        "Genotype"
    ]

    # =========================================================================
    # 80 / 20 SPLIT
    # =========================================================================
    #
    # This reproduces the historical design:
    #
    #     80% training
    #     20% genuine held-out test
    #
    # The split is stratified by genotype.
    # =========================================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # Save split membership
    # -------------------------------------------------------------------------

    split_df = model_df[
        [
            "image_name",
            "Genotype",
        ]
    ].copy()

    split_df[
        "split"
    ] = "test"

    split_df.loc[
        X_train.index,
        "split",
    ] = "train"

    split_df.to_csv(
        cfg.decision_tree_dir
        / "train_test_split.csv",
        index=False,
    )

    # =========================================================================
    # FIXED HISTORICAL TREE
    # =========================================================================

    model_80 = new_fixed_tree()

    model_80.fit(
        X_train,
        y_train,
    )

    # =========================================================================
    # OPTIONAL CROSS-VALIDATION OF THE FIXED MODEL
    # =========================================================================
    #
    # This does NOT select hyperparameters.
    #
    # It simply asks how stable the FIXED historical model is across
    # folds of the 80% training dataset.
    # =========================================================================

    cv_folds = _safe_cv_folds(
        y_train,
        cfg.decision_tree_cv_folds,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_val_score(
        new_fixed_tree(),
        X_train,
        y_train,
        cv=cv,
        scoring=cfg.decision_tree_scoring,
        n_jobs=-1,
    )

    pd.DataFrame(
        {
            "fold": np.arange(
                1,
                len(cv_scores) + 1,
            ),
            "score": cv_scores,
        }
    ).to_csv(
        cfg.decision_tree_dir
        / "fixed_model_cross_validation_scores.csv",
        index=False,
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    ranking = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                model_80.feature_importances_
            ),
        }
    ).sort_values(
        [
            "importance",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    )

    ranking[
        "used_by_tree"
    ] = (
        ranking[
            "importance"
        ]
        > 0
    )

    ranking.to_csv(
        cfg.decision_tree_dir
        / "feature_importance_ranking.csv",
        index=False,
    )

    # =========================================================================
    # GENUINE HELD-OUT 20% TEST
    # =========================================================================

    y_pred = model_80.predict(
        X_test
    )

    labels = list(
        cfg.train_labels
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = {
        "model_type": (
            "fixed_historical_decision_tree"
        ),

        "tree_params": (
            TREE_PARAMS
        ),

        "train_fraction": 0.80,

        "test_fraction": 0.20,

        "random_state": (
            RANDOM_STATE
        ),

        "n_total_complete_cases": int(
            len(model_df)
        ),

        "n_dropped_incomplete": int(
            dropped
        ),

        "n_train": int(
            len(X_train)
        ),

        "n_test": int(
            len(X_test)
        ),

        "cv_folds": int(
            cv_folds
        ),

        "cv_scoring": (
            cfg.decision_tree_scoring
        ),

        "cv_mean_fixed_model": float(
            cv_scores.mean()
        ),

        "cv_sd_fixed_model": (
            float(
                cv_scores.std(
                    ddof=1
                )
            )
            if len(cv_scores) > 1
            else 0.0
        ),

        "test_accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),

        "test_f1_macro": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "test_precision_macro": float(
            precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "test_recall_macro": float(
            recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "test_mcc": float(
            matthews_corrcoef(
                y_test,
                y_pred,
            )
        ),

        "classes": (
            labels
        ),

        "confusion_matrix": (
            cm.tolist()
        ),

        "selected_nonzero_features": (
            ranking.loc[
                ranking[
                    "used_by_tree"
                ],
                "feature",
            ].tolist()
        ),

        "tree_depth_80pct": int(
            model_80.get_depth()
        ),

        "tree_leaves_80pct": int(
            model_80.get_n_leaves()
        ),
    }

    # -------------------------------------------------------------------------
    # Specificity / sensitivity
    # -------------------------------------------------------------------------
    #
    # Assumes cfg.train_labels is:
    #
    #     APP-WT
    #     APP-FIRE
    #
    # Thus:
    #
    # negative = APP-WT
    # positive = APP-FIRE
    # -------------------------------------------------------------------------

    if cm.shape == (
        2,
        2,
    ):

        (
            tn,
            fp,
            fn,
            tp,
        ) = cm.ravel()

        metrics[
            "test_specificity"
        ] = (
            float(
                tn
                / (
                    tn
                    + fp
                )
            )
            if (
                tn
                + fp
            )
            else None
        )

        metrics[
            "test_sensitivity"
        ] = (
            float(
                tp
                / (
                    tp
                    + fn
                )
            )
            if (
                tp
                + fn
            )
            else None
        )

    # -------------------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------------------

    with open(
        cfg.decision_tree_dir
        / "metrics.json",
        "w",
    ) as handle:

        json.dump(
            metrics,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        cfg.decision_tree_dir
        / "classification_report.csv"
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    pd.DataFrame(
        cm,
        index=[
            f"true_{x}"
            for x in labels
        ],
        columns=[
            f"pred_{x}"
            for x in labels
        ],
    ).to_csv(
        cfg.decision_tree_dir
        / "confusion_matrix.csv"
    )

    # =========================================================================
    # HELD-OUT PLAQUE PREDICTIONS
    # =========================================================================

    predictions = pd.DataFrame(
        {
            "image_name": (
                model_df.loc[
                    X_test.index,
                    "image_name",
                ].values
            ),

            "true_label": (
                y_test.values
            ),

            "predicted_label": (
                y_pred
            ),
        }
    )

    probabilities = model_80.predict_proba(
        X_test
    )

    for (
        index,
        class_name,
    ) in enumerate(
        model_80.classes_
    ):

        predictions[
            f"probability_{class_name}"
        ] = probabilities[
            :,
            index,
        ]

    predictions.to_csv(
        cfg.decision_tree_dir
        / "held_out_predictions.csv",
        index=False,
    )

    # =========================================================================
    # HISTORICAL-STYLE: 80%-TRAINED MODEL APPLIED TO ALL PLAQUES
    # =========================================================================
    #
    # IMPORTANT:
    #
    # This is NOT an independent accuracy estimate.
    #
    # 80% of these plaques were used to train the model.
    #
    # It is included because this reproduces the historical style of reporting
    # APP-WT-like / APP-FIRE-like morphology across the complete dataset.
    # =========================================================================

    all_pred = model_80.predict(
        X
    )

    all_probabilities = (
        model_80.predict_proba(
            X
        )
    )

    all_predictions = model_df[
        [
            "image_name",
            "Genotype",
        ]
    ].copy()

    all_predictions[
        "predicted_label"
    ] = all_pred

    for (
        index,
        class_name,
    ) in enumerate(
        model_80.classes_
    ):

        all_predictions[
            f"probability_{class_name}"
        ] = all_probabilities[
            :,
            index,
        ]

    all_predictions[
        "tree_leaf"
    ] = model_80.apply(
        X
    )

    all_predictions.to_csv(
        cfg.decision_tree_dir
        / "historical_style_all_plaques_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Historical-style prediction fractions
    # -------------------------------------------------------------------------

    prediction_counts = pd.crosstab(
        all_predictions[
            "Genotype"
        ],
        all_predictions[
            "predicted_label"
        ],
        margins=True,
    )

    prediction_fractions = pd.crosstab(
        all_predictions[
            "Genotype"
        ],
        all_predictions[
            "predicted_label"
        ],
        normalize="index",
    )

    prediction_counts.to_csv(
        cfg.decision_tree_dir
        / "historical_style_prediction_counts.csv"
    )

    prediction_fractions.to_csv(
        cfg.decision_tree_dir
        / "historical_style_prediction_fractions.csv"
    )

    # =========================================================================
    # SAVE 80%-TRAINED MODEL
    # =========================================================================
    #
    # Useful if you want to apply exactly the 80%-trained historical model
    # to another dataset.
    # =========================================================================

    joblib.dump(
        {
            "model": model_80,

            "features": features,

            "classes": (
                model_80.classes_.tolist()
            ),

            "tree_params": (
                TREE_PARAMS
            ),

            "train_fraction": 0.80,

            "test_fraction": 0.20,

            "random_state": (
                RANDOM_STATE
            ),
        },

        cfg.decision_tree_dir
        / "fixed_historical_dt_80pct.joblib",
    )

    # =========================================================================
    # 100% REFIT FOR TRANSFER
    # =========================================================================
    #
    # Same fixed tree parameters.
    #
    # The ONLY difference is that it is now fitted to all available
    # APP-WT/FIRE plaques.
    #
    # This is the version to use for future external transfer when you want
    # the most information available to define the APP-WT/FIRE morphology
    # boundary.
    # =========================================================================

    final_dt = new_fixed_tree()

    final_dt.fit(
        X,
        y,
    )

    joblib.dump(
        {
            "model": final_dt,

            "features": features,

            "classes": (
                final_dt.classes_.tolist()
            ),

            "tree_params": (
                TREE_PARAMS
            ),

            "train_fraction": 1.00,

            "random_state": (
                RANDOM_STATE
            ),
        },

        cfg.decision_tree_dir
        / "best_dt_combined.joblib",
    )

    # =========================================================================
    # TREE RULES
    # =========================================================================

    with open(
        cfg.decision_tree_dir
        / "tree_rules.txt",
        "w",
    ) as handle:

        handle.write(
            export_text(
                final_dt,
                feature_names=features,
            )
        )

    # Also save rules specifically for the genuine 80% model.
    with open(
        cfg.decision_tree_dir
        / "tree_rules_80pct.txt",
        "w",
    ) as handle:

        handle.write(
            export_text(
                model_80,
                feature_names=features,
            )
        )

    # =========================================================================
    # TREE PLOTS
    # =========================================================================

    # 100% model
    plt.figure(
        figsize=(
            28,
            18,
        )
    )

    plot_tree(
        final_dt,
        feature_names=features,
        class_names=final_dt.classes_,
        filled=True,
        rounded=True,
        fontsize=7,
    )

    plt.tight_layout()

    plt.savefig(
        cfg.decision_tree_dir
        / "decision_tree.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # 80% model
    plt.figure(
        figsize=(
            28,
            18,
        )
    )

    plot_tree(
        model_80,
        feature_names=features,
        class_names=model_80.classes_,
        filled=True,
        rounded=True,
        fontsize=7,
    )

    plt.tight_layout()

    plt.savefig(
        cfg.decision_tree_dir
        / "decision_tree_80pct.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print(
        "\nDecision tree complete"
    )

    print(
        f"  Feature table: {cfg.feature_table_path}"
    )

    print(
        f"  Results:       {cfg.decision_tree_dir}"
    )

    print(
        "\nFixed historical tree:"
    )

    print(
        "  criterion         = entropy"
    )

    print(
        "  max_depth         = 8"
    )

    print(
        "  min_samples_leaf  = 4"
    )

    print(
        "  min_samples_split = 10"
    )

    print(
        "  random_state      = 0"
    )

    print(
        "\nSplit:"
    )

    print(
        f"  Train = {len(X_train)} plaques (80%)"
    )

    print(
        f"  Test  = {len(X_test)} plaques (20%)"
    )

    print(
        "\nFixed-model CV on training 80%:"
    )

    print(
        f"  Mean = {cv_scores.mean():.3f}"
    )

    print(
        f"  SD   = "
        f"{cv_scores.std(ddof=1) if len(cv_scores) > 1 else 0:.3f}"
    )

    print(
        "\nGenuine held-out 20%:"
    )

    print(
        f"  Accuracy = {metrics['test_accuracy']:.3f}"
    )

    print(
        f"  Macro F1 = {metrics['test_f1_macro']:.3f}"
    )

    print(
        f"  MCC      = {metrics['test_mcc']:.3f}"
    )

    print(
        "\nFeatures used by the 80% tree:"
    )

    for feature in metrics[
        "selected_nonzero_features"
    ]:

        print(
            f"    - {feature}"
        )