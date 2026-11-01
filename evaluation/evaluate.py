import csv
import os
import sys

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from training.data_loader import SplitDataGenerator


def main():

    # ---------------------------------------------------------
    # Command-line arguments
    #
    # Usage:
    # python -m evaluation.evaluate B1 R0_spatial 123
    #
    # benchmark = B1
    # method = R0_spatial
    # training_seed = 123
    #
    # The validation split ALWAYS remains seed 42.
    # ---------------------------------------------------------

    benchmark = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "B1"
    )

    method = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "R0_spatial"
    )

    training_seed = (
        int(sys.argv[3])
        if len(sys.argv) > 3
        else 42
    )

    # Fixed data split
    split_seed = 42

    block_size = None
    dct_k = None

    # ---------------------------------------------------------
    # Method selection
    # ---------------------------------------------------------

    if method == "R0_spatial":

        if training_seed == 42:
            model_path = (
                f"models/saved/"
                f"{benchmark}_baseline_seed42.keras"
            )
        else:
            model_path = (
                f"models/saved/"
                f"{benchmark}_baseline_"
                f"trainseed{training_seed}_"
                f"splitseed{split_seed}.keras"
            )

        preprocessing = "spatial"

    elif method == "R1_dct":

        if training_seed == 42:
            model_path = (
                f"models/saved/"
                f"{benchmark}_R1_dct_seed42.keras"
            )
        else:
            model_path = (
                f"models/saved/"
                f"{benchmark}_R1_dct_"
                f"trainseed{training_seed}_"
                f"splitseed{split_seed}.keras"
            )

        preprocessing = "dct"

    elif method.startswith("dct_zigzag_k"):

        dct_k = int(
            method.replace(
                "dct_zigzag_k",
                "",
            )
        )

        if training_seed == 42:
            model_path = (
                f"models/saved/"
                f"{benchmark}_dct_zigzag_"
                f"k{dct_k}_seed42.keras"
            )
        else:
            model_path = (
                f"models/saved/"
                f"{benchmark}_dct_zigzag_"
                f"k{dct_k}_"
                f"trainseed{training_seed}_"
                f"splitseed{split_seed}.keras"
            )

        preprocessing = "dct_zigzag"

    elif method.startswith("dct_rowmajor_k"):

        dct_k = int(
            method.replace(
                "dct_rowmajor_k",
                "",
            )
        )

        if training_seed == 42:
            model_path = (
                f"models/saved/"
                f"{benchmark}_dct_rowmajor_"
                f"k{dct_k}_seed42.keras"
            )
        else:
            model_path = (
                f"models/saved/"
                f"{benchmark}_dct_rowmajor_"
                f"k{dct_k}_"
                f"trainseed{training_seed}_"
                f"splitseed{split_seed}.keras"
            )

        preprocessing = "dct_rowmajor"

    elif method.startswith("dct_block_bs"):

        # Expected:
        # dct_block_bs16_k4096

        parts = method.split("_")

        block_size = int(
            parts[2].replace(
                "bs",
                "",
            )
        )

        dct_k = int(
            parts[3].replace(
                "k",
                "",
            )
        )

        if training_seed == 42:
            model_path = (
                f"models/saved/"
                f"{benchmark}_dct_block_bs"
                f"{block_size}_k{dct_k}_seed42.keras"
            )
        else:
            model_path = (
                f"models/saved/"
                f"{benchmark}_dct_block_bs"
                f"{block_size}_k{dct_k}_"
                f"trainseed{training_seed}_"
                f"splitseed{split_seed}.keras"
            )

        preprocessing = "dct_block"

    else:

        raise ValueError(
            f"Unknown method '{method}'. "
            "Expected 'R0_spatial', "
            "'R1_dct', "
            "'dct_zigzag_k<value>', "
            "'dct_rowmajor_k<value>', "
            "or 'dct_block_bs<block>_k<value>'."
        )

    # ---------------------------------------------------------
    # Load FIXED validation split
    # ---------------------------------------------------------

    split_file = (
        f"splits/"
        f"{benchmark}_split_seed{split_seed}.npz"
    )

    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    data = np.load(split_file)

    paths = data["paths"]
    labels = data["labels"]
    val_idx = data["val_idx"]

    # ---------------------------------------------------------
    # Validation generator
    # ---------------------------------------------------------

    val_generator = SplitDataGenerator(
        paths,
        labels,
        val_idx,
        batch_size=32,
        shuffle=False,
        preprocessing=preprocessing,
        dct_k=dct_k,
        block_size=block_size,
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    # ---------------------------------------------------------
    # Print experiment information
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)

    print(
        f"Benchmark:       {benchmark}"
    )

    print(
        f"Method:          {method}"
    )

    print(
        f"Training seed:   {training_seed}"
    )

    print(
        f"Split seed:      {split_seed}"
    )

    print(
        f"Preprocessing:   {preprocessing}"
    )

    if dct_k is not None:
        print(
            f"DCT K:           {dct_k}"
        )

    if block_size is not None:
        print(
            f"Block size:      "
            f"{block_size}x{block_size}"
        )

    print(
        f"Split:           {split_file}"
    )

    print(
        f"Validation:      {len(val_idx)}"
    )

    print(
        f"Model:           {model_path}"
    )

    print(
        f"Parameters:      "
        f"{model.count_params()}"
    )

    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    probabilities = model.predict(
        val_generator,
        verbose=1,
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    y_true = (
        labels[val_idx]
        .astype(int)
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            predictions,
        )
    )

    sensitivity = recall

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    print(
        f"TP:                 {tp}"
    )

    print(
        f"TN:                 {tn}"
    )

    print(
        f"FP:                 {fp}"
    )

    print(
        f"FN:                 {fn}"
    )

    print(
        f"Sensitivity/Recall: "
        f"{sensitivity:.4f}"
    )

    print(
        f"Specificity:        "
        f"{specificity:.4f}"
    )

    print(
        f"Precision:          "
        f"{precision:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{f1:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True,
    )

    metrics_file = (
        "results/metrics.csv"
    )

    file_exists = os.path.exists(
        metrics_file
    )

    row = {
        "benchmark": benchmark,
        "method": method,
        "training_seed": training_seed,
        "split_seed": split_seed,
        "block_size": (
            block_size
            if block_size is not None
            else ""
        ),
        "dct_k": (
            dct_k
            if dct_k is not None
            else ""
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }

    fieldnames = list(
        row.keys()
    )

    with open(
        metrics_file,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print()
    print(
        f"Saved metrics: "
        f"{metrics_file}"
    )


if __name__ == "__main__":
    main()
