import tensorflow as tf

import numpy as np
import os
import time

from LyricModel import LyricBot
from Step import Step


def clean(fn):
    print('Cleaning:' + fn)
    f = open(fn, 'r')
    lines = f.read()
    f.close()
    r = ['Translations',
         'Türkçe',
         'Español',
         'Русский',
         'Português',
         'Italiano',
         'Deutsch',
         'Français',
         'NederlandsAll',
         '\u205f',
         '中',
         '文',
         'ا',
         'ت',
         'ج',
         'ر',
         'س',
         'ف',
         'م',
         'ه',
         'ی',
         '\u2005',
         '\u200b']
    for i in r:
        lines.replace(i, "")

    f = open(f"{fn}2.txt", 'w')
    f.write(lines)
    f.close()


def get_text():
    t = open("jw.txt", 'rb').read().decode('utf-8')
    print(f'Length of text: {len(t)}')
    return t


def make_vocab(t):
    v = sorted(set(t))
    print(f'{len(v)} unique characters')
    return v


def text_from_ids(ids1):
    return tf.strings.reduce_join(chars_from_ids(ids1), axis=-1)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


text = get_text()
vocab = make_vocab(text)

chars = tf.strings.unicode_split(text, 'UTF-8')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
chars = chars_from_ids(ids)

tf.strings.reduce_join(chars, axis=-1).numpy()


all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


seq_length = 100

sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)


BATCH_SIZE = 64


# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(buffer_size=BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

model = LyricBot(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)
model.summary()
print(f"Input:\n{text_from_ids(input_example_batch[0]).numpy()}\n")
print(f'Next char predictions:\n{text_from_ids(sampled_indices).numpy()}')

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print(f'Prediction loss:\n{example_batch_mean_loss.shape}')
print(f"Mean loss: {example_batch_mean_loss}")

tf.exp(example_batch_mean_loss).numpy()

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
)

EPOCHS = 20

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


stepper = Step(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['I still'])
result = [next_char]

for n in range(1000):
    next_char, states = stepper.generate_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)


