import os
import sys
import yaml
import numpy as np
import tensorflow as tf

from models.baseline_cnn import build_baseline
from training.data_loader import SplitDataGenerator


def main():
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "B1"
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    with open("configs/baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["split"]["seed"]
    k = 4096

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
            f"Expected one of: {list(epochs_by_benchmark.keys())}"
        )

    if block_size <= 0 or 224 % block_size != 0:
        raise ValueError(
            f"block_size must be a positive divisor of 224, "
            f"got {block_size}"
        )

    epochs = epochs_by_benchmark[benchmark]

    np.random.seed(seed)
    tf.random.set_seed(seed)

    split_file = f"splits/{benchmark}_split_seed{seed}.npz"
    data = np.load(split_file)

    paths = data["paths"]
    labels = data["labels"]

    train_generator = SplitDataGenerator(
        paths,
        labels,
        data["train_idx"],
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle_train"],
        preprocessing="dct_block",
        dct_k=k,
        block_size=block_size,
    )

    val_generator = SplitDataGenerator(
        paths,
        labels,
        data["val_idx"],
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle_validation"],
        preprocessing="dct_block",
        dct_k=k,
        block_size=block_size,
    )

    model = build_baseline()

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Benchmark: {benchmark}")
    print("Preprocessing: Block DCT + Zig-Zag")
    print(f"Block size: {block_size}x{block_size}")
    print(f"Total coefficients: {k}")
    print(f"Split: {split_file}")
    print(f"Train samples: {len(data['train_idx'])}")
    print(f"Validation samples: {len(data['val_idx'])}")
    print(f"Parameters: {model.count_params()}")
    print(f"Epochs: {epochs}")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
    )

    os.makedirs("models/saved", exist_ok=True)

    model_path = (
        f"models/saved/"
        f"{benchmark}_dct_block_bs{block_size}_k{k}_seed{seed}.keras"
    )

    model.save(model_path)

    history_path = (
        f"results/"
        f"{benchmark}_dct_block_bs{block_size}_k{k}_seed{seed}_history.npz"
    )

    np.savez(
        history_path,
        **history.history,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
