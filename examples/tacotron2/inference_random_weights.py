"""Run TFTacotron2 inference on random token ids of varying length, with random weights.

Built from test/test_tacotron2.py: a default Tacotron2Config, no checkpoint and no dataset, so it
runs without LJSpeech. maximum_iterations is held at one small constant, so decoding stays short
and the Python-valued arguments are the same on every call; only the input length varies.
"""
import time

import numpy as np
import tensorflow as tf

from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.models import TFTacotron2

MAX_ITERATIONS = 50
LENGTHS = [12, 17, 23, 31, 40, 52, 67, 85]

tf.random.set_seed(0)
config = Tacotron2Config(n_speakers=1, reduction_factor=1)
model = TFTacotron2(config, training=False)
model._build()

start = time.time()
for length in LENGTHS:
    input_ids = tf.random.uniform([1, length], minval=1, maxval=config.vocab_size, dtype=tf.int32)
    outputs = model.inference(input_ids, tf.constant([length], tf.int32), tf.constant([0], tf.int32),
                              maximum_iterations=MAX_ITERATIONS)
    print("length %3d -> mel %s" % (length, tuple(outputs[1].shape)))
print("inference traces: %d" % model.inference.experimental_get_tracing_count())
print("elapsed %.1fs" % (time.time() - start))
