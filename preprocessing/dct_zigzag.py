import tensorflow as tf

from preprocessing.zigzag import zigzag_indices


def load_dct_zigzag_image(path, k):
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

    # Keep first K coefficients in zig-zag order
    indices = zigzag_indices(224)[:k]

    mask = tf.scatter_nd(
        indices,
        tf.ones((k,), dtype=dct.dtype),
        shape=(224, 224),
    )

    dct = dct * mask

    # Same magnitude compression as R1
    dct = tf.math.abs(dct)
    dct = tf.math.log1p(dct)

    # Same per-image normalization as R1
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
