import os
import sys
import yaml
import numpy as np
import tensorflow as tf

from models.baseline_cnn import build_baseline
from training.data_loader import SplitDataGenerator


def main():

    # ---------------------------------------------------------
    # Command-line arguments
    #
    # Usage:
    # python -m training.train_dct <benchmark> <k> <training_seed>
    # ---------------------------------------------------------

    benchmark = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "B1"
    )

    if len(sys.argv) < 3:
        raise ValueError(
            "Usage: "
            "python -m training.train_dct "
            "<benchmark> <k> <training_seed>"
        )

    k = int(sys.argv[2])

    training_seed = (
        int(sys.argv[3])
        if len(sys.argv) > 3
        else 42
    )

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open(
        "configs/baseline.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    # IMPORTANT:
    # The train/validation split stays FIXED at seed 42.
    split_seed = config["split"]["seed"]

    # ---------------------------------------------------------
    # Validate K
    # ---------------------------------------------------------

    if k <= 0 or k > 224 * 224:
        raise ValueError(
            f"k must be between 1 and {224 * 224}, "
            f"got {k}"
        )

    # ---------------------------------------------------------
    # Epoch schedule
    # ---------------------------------------------------------

    epochs_by_benchmark = {
        "B1": 5,
        "B2": 5,
        "B3": 10,
        "B4": 10,
        "B5": 10,
    }

    if benchmark not in epochs_by_benchmark:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Expected one of: "
            f"{list(epochs_by_benchmark.keys())}"
        )

    epochs = epochs_by_benchmark[benchmark]

    # ---------------------------------------------------------
    # Set TRAINING random seed
    # ---------------------------------------------------------

    np.random.seed(training_seed)
    tf.random.set_seed(training_seed)

    # ---------------------------------------------------------
    # Load FIXED split
    # ---------------------------------------------------------

    split_file = (
        f"splits/"
        f"{benchmark}_split_seed{split_seed}.npz"
    )

    data = np.load(split_file)

    paths = data["paths"]
    labels = data["labels"]

    # ---------------------------------------------------------
    # Training generator
    # ---------------------------------------------------------

    train_generator = SplitDataGenerator(
        paths,
        labels,
        data["train_idx"],
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle_train"],
        preprocessing="dct_zigzag",
        dct_k=k,
    )

    # ---------------------------------------------------------
    # Validation generator
    # ---------------------------------------------------------

    val_generator = SplitDataGenerator(
        paths,
        labels,
        data["val_idx"],
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle_validation"],
        preprocessing="dct_zigzag",
        dct_k=k,
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    model = build_baseline()

    # ---------------------------------------------------------
    # Compile
    # ---------------------------------------------------------

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # ---------------------------------------------------------
    # Experiment information
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Multi-seed DCT training")
    print("=" * 60)

    print(
        f"Benchmark:       {benchmark}"
    )

    print(
        f"Preprocessing:   DCT + Zig-Zag"
    )

    print(
        f"Coefficients:    {k}"
    )

    print(
        f"Training seed:   {training_seed}"
    )

    print(
        f"Split seed:      {split_seed}"
    )

    print(
        f"Split:           {split_file}"
    )

    print(
        f"Train samples:   "
        f"{len(data['train_idx'])}"
    )

    print(
        f"Validation:      "
        f"{len(data['val_idx'])}"
    )

    print(
        f"Parameters:      "
        f"{model.count_params()}"
    )

    print(
        f"Epochs:          {epochs}"
    )

    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    os.makedirs(
        "models/saved",
        exist_ok=True,
    )

    model_path = (
        f"models/saved/"
        f"{benchmark}_dct_zigzag_k{k}_"
        f"trainseed{training_seed}_"
        f"splitseed{split_seed}.keras"
    )

    model.save(model_path)

    # ---------------------------------------------------------
    # Save history
    # ---------------------------------------------------------

    history_path = (
        f"results/"
        f"{benchmark}_dct_zigzag_k{k}_"
        f"trainseed{training_seed}_"
        f"splitseed{split_seed}_"
        f"history.npz"
    )

    np.savez(
        history_path,
        **history.history,
    )

    print()
    print(
        f"Saved model: "
        f"{model_path}"
    )

    print(
        f"Saved history: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()
