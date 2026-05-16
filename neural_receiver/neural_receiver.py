"""DT-augmented neural OFDM receiver — Conv2D-ResNet architecture.

Input:  (batch, 192, 14, 6)  — resource grid with DT prior
Output: (batch, 192, 14, 2)  — per-RE LLRs for QPSK (2 bits)

The 14 columns are: P0 (idx 0), D0-D11 (idx 1-12), P1 (idx 13).
Loss is masked to data positions only (columns 1-12).

Input channels:
  0-1: Re(Y), Im(Y)         — received signal
  2-3: Re(H_dt), Im(H_dt)   — DT channel prior (broadcast over time)
  4:   pilot_mask            — 1 at P0/P1 columns, 0 at data
  5:   N0                    — noise variance (scalar broadcast)
"""

import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.bn2(self.conv2(x), training=training)
        return tf.nn.relu(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self._filters, 'kernel_size': self._kernel_size})
        return config


@keras.saving.register_keras_serializable()
class NeuralOFDMReceiver(tf.keras.Model):
    """DT-augmented neural OFDM receiver.

    Args:
        num_res_blocks: Number of residual blocks.
        filters: Conv2D filter count per layer.
        kernel_size: Spatial kernel size.
        use_dt_prior: If False, drops H_dt input channels (ablation).
    """

    def __init__(self, num_res_blocks=3, filters=64,
                 kernel_size=(3, 3), use_dt_prior=True, n_bits=2, **kwargs):
        super().__init__(**kwargs)
        self.num_res_blocks = num_res_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_dt_prior = use_dt_prior
        self.n_bits = n_bits
        c_in = 8 if use_dt_prior else 4

        self.input_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', activation='relu')

        self.res_blocks = [
            ResBlock(filters, kernel_size, name=f'res_{i}')
            for i in range(num_res_blocks)
        ]

        self.output_conv = tf.keras.layers.Conv2D(
            n_bits, (1, 1), padding='same')

    def call(self, inputs, training=False):
        x = self.input_conv(inputs)
        for block in self.res_blocks:
            x = block(x, training=training)
        return self.output_conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_res_blocks': self.num_res_blocks,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'use_dt_prior': self.use_dt_prior,
            'n_bits': self.n_bits,
        })
        return config


def build_data_mask():
    """Binary mask: 1 at data positions (sym 1-12), 0 at pilots (sym 0, 13).

    Returns: tf.Tensor shape (1, 1, 14, 1), broadcastable to (B, 192, 14, 2).
    """
    mask = tf.concat([
        tf.zeros([1, 1, 1, 1]),     # P0
        tf.ones([1, 1, 12, 1]),     # D0-D11
        tf.zeros([1, 1, 1, 1]),     # P1
    ], axis=2)
    return mask


def masked_bce_loss(y_true, y_pred, data_mask):
    """BCE loss computed only at data RE positions."""
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    masked = bce * data_mask
    n_data = tf.reduce_sum(data_mask) * tf.cast(
        tf.shape(y_true)[0] * tf.shape(y_true)[1] * tf.shape(y_true)[3],
        tf.float32)
    return tf.reduce_sum(masked) / n_data


def logits_to_bits(logits):
    """Hard-decision from BCE logits: logit > 0 => bit 1, logit <= 0 => bit 0."""
    return tf.cast(logits > 0, tf.int32)


if __name__ == '__main__':
    model = NeuralOFDMReceiver(num_res_blocks=3, filters=64, use_dt_prior=True)
    dummy = tf.random.normal((4, 192, 14, 6))
    out = model(dummy, training=False)
    print(f'QPSK+DT output shape: {out.shape}')
    model.summary()
