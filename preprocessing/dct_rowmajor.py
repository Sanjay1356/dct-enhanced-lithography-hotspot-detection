import tensorflow as tf


def rowmajor_indices(n):
    rows = tf.repeat(
        tf.range(n, dtype=tf.int32),
        n,
    )

    cols = tf.tile(
        tf.range(n, dtype=tf.int32),
        [n],
    )

    return tf.stack([rows, cols], axis=1)


def load_dct_rowmajor_image(path, k):
    image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224),
        color_mode="grayscale",
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0

    # 2-D orthonormal DCT
    dct = tf.signal.dct(
        image[:, :, 0],
        type=2,
        norm="ortho",
    )

    dct = tf.transpose(
        tf.signal.dct(
            tf.transpose(dct),
            type=2,
            norm="ortho",
        )
    )

    # Keep first K coefficients in ROW-MAJOR order
    indices = rowmajor_indices(224)[:k]

    mask = tf.scatter_nd(
        indices,
        tf.ones((k,), dtype=dct.dtype),
        shape=(224, 224),
    )

    dct = dct * mask

    # Same processing as zig-zag implementation
    dct = tf.math.abs(dct)
    dct = tf.math.log1p(dct)

    # Per-image min-max normalization
    dct_min = tf.reduce_min(dct)
    dct_max = tf.reduce_max(dct)

    dct = (dct - dct_min) / (
        dct_max - dct_min + 1e-8
    )

    # Match the 3-channel CNN input
    dct = tf.stack(
        [dct, dct, dct],
        axis=-1,
    )

    return dct
