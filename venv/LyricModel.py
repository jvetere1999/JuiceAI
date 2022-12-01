import tensorflow as tf


class LyricBot(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense1 = tf.keras.layers.Dense(vocab_size)
        self.dense2 = tf.keras.layers.Dense(vocab_size)
        self.dense3 = tf.keras.layers.Dense(vocab_size)
        self.dense4 = tf.keras.layers.Dense(vocab_size)
        self.dense5 = tf.keras.layers.Dense(vocab_size)
        self.dense6 = tf.keras.layers.Dense(vocab_size)
        self.dense7 = tf.keras.layers.Dense(vocab_size)
        self.dense8 = tf.keras.layers.Dense(vocab_size)
        self.dense9 = tf.keras.layers.Dense(vocab_size)
        self.dense10 = tf.keras.layers.Dense(vocab_size)
        self.dense11 = tf.keras.layers.Dense(vocab_size)
        self.dense12 = tf.keras.layers.Dense(vocab_size)
        self.dense13 = tf.keras.layers.Dense(vocab_size)
        self.dense14 = tf.keras.layers.Dense(vocab_size)
        self.dense15 = tf.keras.layers.Dense(vocab_size)
        self.dense16 = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.dense4(x, training=training)
        x = self.dense5(x, training=training)
        x = self.dense6(x, training=training)
        x = self.dense7(x, training=training)
        x = self.dense8(x, training=training)
        x = self.dense9(x, training=training)
        x = self.dense10(x, training=training)
        x = self.dense11(x, training=training)
        x = self.dense12(x, training=training)
        x = self.dense13(x, training=training)
        x = self.dense14(x, training=training)
        x = self.dense15(x, training=training)
        x = self.dense16(x, training=training)
        if return_state:
            return x, states
        else:
            return x
