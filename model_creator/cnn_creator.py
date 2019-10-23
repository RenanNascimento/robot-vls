from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf

from cnn_model import cnn_model_fn
from load_dataset import load_dataset

tf.logging.set_verbosity(tf.logging.INFO)

# Load training and eval data
train_data, train_labels, eval_data, eval_labels = load_dataset()

# Create the Estimator
fiducial_marker_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=os.environ.get('CNN_MODEL'))

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=250,
    num_epochs=None,
    shuffle=True)

# train 1000 step and display the probabilties
fiducial_marker_classifier.train(input_fn=train_input_fn, steps=1000)

# Evaluate the Model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = fiducial_marker_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)