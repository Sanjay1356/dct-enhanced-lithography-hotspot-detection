import numpy as np
import tensorflow as tf

from preprocessing.dct import load_dct_image
from preprocessing.dct_zigzag import load_dct_zigzag_image
from preprocessing.dct_rowmajor import load_dct_rowmajor_image
from preprocessing.dct_block import load_block_dct_image


class SplitDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        paths,
        labels,
        indices,
        batch_size=32,
        shuffle=False,
        preprocessing="spatial",
        dct_k=None,
        block_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if preprocessing not in (
            "spatial",
            "dct",
            "dct_zigzag",
            "dct_rowmajor",
            "dct_block",
        ):
            raise ValueError(
                f"Unknown preprocessing '{preprocessing}'. "
                "Expected 'spatial', 'dct', "
                "'dct_zigzag', 'dct_rowmajor', "
                "or 'dct_block'."
            )

        if preprocessing in (
            "dct_zigzag",
            "dct_rowmajor",
            "dct_block",
        ):
            if dct_k is None:
                raise ValueError(
                    f"dct_k must be provided when using "
                    f"'{preprocessing}'."
                )

            if dct_k <= 0 or dct_k > 224 * 224:
                raise ValueError(
                    f"dct_k must be between 1 and {224 * 224}, "
                    f"got {dct_k}"
                )

        if preprocessing == "dct_block":
            if block_size is None:
                raise ValueError(
                    "block_size must be provided when using "
                    "'dct_block'."
                )

            if block_size <= 0 or 224 % block_size != 0:
                raise ValueError(
                    "block_size must be a positive divisor of 224, "
                    f"got {block_size}"
                )

        self.paths = paths
        self.labels = labels
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.dct_k = dct_k
        self.block_size = block_size

        self.on_epoch_end()

    def __len__(self):
        return int(
            np.ceil(
                len(self.indices) / self.batch_size
            )
        )

    def __getitem__(self, batch_index):
        batch_indices = self.indices[
            batch_index * self.batch_size:
            (batch_index + 1) * self.batch_size
        ]

        images = []
        labels = []

        for idx in batch_indices:

            if self.preprocessing == "spatial":
                image = tf.keras.utils.img_to_array(
                    tf.keras.utils.load_img(
                        self.paths[idx],
                        target_size=(224, 224),
                    )
                )

                image = image / 255.0

            elif self.preprocessing == "dct":
                image = load_dct_image(
                    self.paths[idx]
                )

            elif self.preprocessing == "dct_zigzag":
                image = load_dct_zigzag_image(
                    self.paths[idx],
                    self.dct_k,
                )

            elif self.preprocessing == "dct_rowmajor":
                image = load_dct_rowmajor_image(
                    self.paths[idx],
                    self.dct_k,
                )

            elif self.preprocessing == "dct_block":
                image = load_block_dct_image(
                    self.paths[idx],
                    self.block_size,
                    self.dct_k,
                )

            images.append(image)
            labels.append(self.labels[idx])

        return (
            np.asarray(
                images,
                dtype=np.float32,
            ),
            np.asarray(
                labels,
                dtype=np.float32,
            ),
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
