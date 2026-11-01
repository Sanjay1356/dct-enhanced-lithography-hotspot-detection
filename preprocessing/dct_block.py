import tensorflow as tf


def zigzag_indices(n):
    indices = []

    for s in range(2 * n - 1):
        if s % 2 == 0:
            r_start = min(s, n - 1)
            r_end = max(-1, s - n)
            for r in range(r_start, r_end, -1):
                c = s - r
                indices.append((r, c))
        else:
            r_start = max(0, s - n + 1)
            r_end = min(s, n - 1)
            for r in range(r_start, r_end + 1):
                c = s - r
                indices.append((r, c))

    return tf.constant(indices, dtype=tf.int32)


def load_block_dct_image(path, block_size, total_k=4096):
    image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224),
        color_mode="grayscale",
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = image[:, :, 0]

    # Number of blocks along each dimension
    blocks_per_dim = 224 // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    # Distribute total coefficient budget equally
    k_per_block = total_k // num_blocks

    # Never exceed coefficients available in a block
    k_per_block = min(
        k_per_block,
        block_size * block_size,
    )

    zz = zigzag_indices(block_size)[:k_per_block]

    output = tf.zeros((224, 224), dtype=tf.float32)

    for br in range(blocks_per_dim):
        for bc in range(blocks_per_dim):

            r0 = br * block_size
            c0 = bc * block_size

            block = image[
                r0:r0 + block_size,
                c0:c0 + block_size,
            ]

            dct = tf.signal.dct(
                block,
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

            mask = tf.scatter_nd(
                zz,
                tf.ones(
                    (k_per_block,),
                    dtype=dct.dtype,
                ),
                shape=(block_size, block_size),
            )

            dct = dct * mask

            dct = tf.math.abs(dct)
            dct = tf.math.log1p(dct)

            output = tf.tensor_scatter_nd_update(
                output,
                tf.reshape(
                    tf.stack(
                        tf.meshgrid(
                            tf.range(r0, r0 + block_size),
                            tf.range(c0, c0 + block_size),
                            indexing="ij",
                        ),
                        axis=-1,
                    ),
                    [-1, 2],
                ),
                tf.reshape(dct, [-1]),
            )

    dct_min = tf.reduce_min(output)
    dct_max = tf.reduce_max(output)

    output = (output - dct_min) / (
        dct_max - dct_min + 1e-8
    )

    return tf.stack(
        [output, output, output],
        axis=-1,
    )
