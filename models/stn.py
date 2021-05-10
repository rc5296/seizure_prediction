import tensorflow as tf
import tensorflow_probability as tfp


class SequenceTransformer(tf.keras.layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)

        self.norm1 = tf.keras.layers.BatchNormalization(axis=1, name='stn_normal1')
        self.conv1 = tf.keras.layers.Conv1D(16, 20, strides=1, activation='relu', data_format='channels_first', name='stn_conv1')
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=10, strides=10, data_format='channels_first')
        self.conv2 = tf.keras.layers.Conv1D(32, 20, activation='relu', data_format='channels_first', name='stn_conv2')
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=10, strides=10, data_format='channels_first')

        self.flat = tf.keras.layers.Flatten()
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', name='stn_dens1')
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(4, activation=None,
                                         bias_initializer=tf.keras.initializers.constant([1.0, 0, 1.0, 0]),
                                         kernel_initializer='zeros', name='stn_dens2')

    def call(self, inputs):
        # x should be the shape (# clips, channels, samples)
        x = self.norm1(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        theta_phi = self.fc2(x)

        x_hat = self.temporal_transform(inputs, theta_phi[:, :2])
        # print('x_hat shape={}'.format(x_hat.shape))
        x_hat = self.magnitude_transform(x_hat, theta_phi[:, 2:])
        # print('x_hat shape={}'.format(x_hat.shape))
        # tf.print(theta_phi, [theta_phi], 'THis is theta_phi')
        return [x_hat, theta_phi]

    def temporal_transform(self, x, theta):
        shape = tf.shape(x)
        idx = tf.range(x.shape[2], dtype=tf.float32)
        idx = tf.stack([idx, tf.ones_like(idx)], axis=0)
        new_idx = tf.matmul(theta, idx)
        # print('new_idx shape={}'.format(new_idx.shape))
        channel_last = tf.transpose(x, perm=[0, 2, 1])

        x_hat = tfp.math.batch_interp_regular_1d_grid(x=new_idx,
                                                      x_ref_min=0.,
                                                      x_ref_max=float(x.shape[2]) - 1.,
                                                      y_ref=channel_last,
                                                      axis=1,
                                                      )

        return tf.transpose(x_hat, perm=[0, 2, 1])

    def magnitude_transform(self, x, phi):
        phi_1 = tf.reshape(phi[:, 1], [tf.shape(phi)[0], 1, 1])
        phi_1 = tf.tile(phi_1, [1, tf.shape(x)[1], 1])
        phi_0 = tf.reshape(phi[:, 0], [tf.shape(phi)[0], 1, 1])
        phi_0 = tf.tile(phi_0, [1, tf.shape(x)[1], 1])

        x_hat = tf.math.multiply(x, phi_0)
        x_hat = tf.math.add(x_hat, phi_1)

        return x_hat

class STFT(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        stft = tf.abs(tf.signal.stft(x,
                              frame_length=512, frame_step=256,
                              window_fn=tf.signal.hann_window))

        # TODO: remove powerline noise

        stft = tf.math.log(stft + 1e-6)
        # stft = tf.where(tf.math.is_nan(stft), tf.zeros_like(stft), stft)

        return tf.transpose(stft, perm=[0, 1, 3, 2])