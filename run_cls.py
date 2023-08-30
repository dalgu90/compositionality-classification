# coding=utf-8
# Copyright 2021 The Compositional Classification Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/python
"""Main Python file to run the classification experiments."""

import os
import json
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
from modules import datasets
from modules import hyperparameters
from modules import losses
from modules import models
from modules import optimizers
from sklearn import metrics
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', 'lstm',
                  ['lstm', 'transformer', 'relative_transformer'],
                  'Model architecture.')
flags.DEFINE_string('data_dir', 'data', 'Path to the dataset root.')
flags.DEFINE_bool('do_train', False, 'Whether to run training.')
flags.DEFINE_enum('dataset', 'test', ['train', 'train_holdout', 'dev', 'test'],
                  'Dataset split to use when do_train=False.')
flags.DEFINE_string('output_dir', 'exp_lstm', 'Output directory to save '
                    'log and checkpoints.')
flags.DEFINE_integer('train_steps', 10000, 'Number of steps to train.')
flags.DEFINE_integer('checkpoint_iter', 1000, 'Steps per checkpoint save.')
flags.DEFINE_integer('eval_iter', 500, 'Steps per validation.')
flags.DEFINE_integer('display_iter', 100, 'Steps per print.')

flags.register_validator('data_dir', os.path.exists, 'Dataset not found.')



def get_epoch_result(model, dataset, loss_fn, return_f1_auc=False):
  """Evaluates loss and accuracy of the model on the dataset."""
  total_logits, total_labels = [], []
  for batch_inputs, batch_labels in dataset:
    batch_logits = model(batch_inputs)
    total_logits.append(batch_logits)
    total_labels.append(batch_labels)
  total_logits = tf.concat(total_logits, -1)
  total_labels = tf.concat(total_labels, -1)
  total_preds = tf.cast(total_logits > 0.0, tf.int32)
  loss = tf.reduce_mean(loss_fn(total_labels, total_logits))
  acc = tf.reduce_mean(tf.cast(total_labels == total_preds, tf.float32))
  if not return_f1_auc:
    return loss, acc
  else:
    tp = tf.reduce_sum(tf.cast((total_labels==1) & (total_preds==1), tf.int32))
    fp = tf.reduce_sum(tf.cast((total_labels==0) & (total_preds==1), tf.int32))
    fn = tf.reduce_sum(tf.cast((total_labels==1) & (total_preds==0), tf.int32))
    f1 = 2 * tp / (2 * tp + fp + fn)
    total_probs = 1.0 / (1.0 + tf.exp(-total_logits))
    auc = metrics.roc_auc_score(total_labels, total_probs)
    return loss, acc, f1, auc


def main(argv):
  del argv  # unused

  for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

  # Loads hyper-parameters.
  if FLAGS.model == 'lstm':
    hparams = hyperparameters.lstm_model_hparams()
  elif FLAGS.model == 'transformer':
    hparams = hyperparameters.transformer_model_hparams()
  elif FLAGS.model == 'relative_transformer':
    hparams = hyperparameters.relative_transformer_model_hparams()

  # Loads datasets
  if FLAGS.parse_tree_input:
    dataset_fn = datasets.load_cls_mask_dataset
  elif FLAGS.model != 'relative_transformer':
    dataset_fn = datasets.load_cls_dataset
  else:
    dataset_fn = datasets.load_cls_nomask_dataset

  if FLAGS.do_train:
    dataset_train = dataset_fn(hparams, FLAGS.data_dir, name='train',
                               shuffle_repeat=True)
    dataset_val = dataset_fn(hparams, FLAGS.data_dir, name='dev')
    data_iter_train = iter(dataset_train)
  else:
    dataset_test = dataset_fn(hparams, FLAGS.data_dir, name=FLAGS.dataset)
  with open(os.path.join(FLAGS.data_dir, 'vocab.txt'), 'r') as fd:
    hparams.vocab_size = len(fd.readlines())

  # Builds model.
  if FLAGS.model == 'lstm':
    model = models.build_lstm_model(hparams)
  elif FLAGS.model == 'transformer':
    hparams.pad_index = datasets.PAD_TOKEN_ID
    hparams.sep_index = datasets.SEP_TOKEN_ID
    model = models.build_transformer_model(hparams)
  elif FLAGS.model == 'relative_transformer':
    model = models.build_relative_transformer_model(hparams)

  # Define loss and optimizer.
  loss_fn = losses.get_weighted_binary_cross_entropy_fn(hparams)
  optimizer = optimizers.get_optimizer(hparams)

  # Create checkpoint manager and initialize.
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  init_step = 0
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint, directory=FLAGS.output_dir, max_to_keep=5)
  if ckpt_manager.latest_checkpoint:
    ckpt_path = ckpt_manager.latest_checkpoint
    logging.info('Load model checkpoint from %s', ckpt_path)
    checkpoint.restore(ckpt_path)
    init_step = int(ckpt_path[ckpt_path.rfind('-') + 1:])

  if FLAGS.do_train:
    # Training model
    for step in range(init_step, FLAGS.train_steps):
      batch_inputs, batch_labels = next(data_iter_train)
      with tf.GradientTape() as tape:
        batch_logits = model(batch_inputs, training=True)
        loss = tf.reduce_mean(loss_fn(batch_labels, batch_logits))
      acc = tf.reduce_mean(
          tf.cast(batch_labels == tf.cast(batch_logits > 0.0, tf.int32),
                  tf.float32))
      # Retrives lr before the optimizer step increases.
      lr = optimizers.get_lr(optimizer)
      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      if step % FLAGS.display_iter == 0 or step == FLAGS.train_steps - 1:
        logging.info('(Train) step %6i, loss=%.6f, acc=%.4f, lr=%.5f', step,
                     loss, acc, lr)

      # Testing model on validation set
      if step % FLAGS.eval_iter == 0 or step == FLAGS.train_steps - 1:
        loss, acc = get_epoch_result(model, dataset_val, loss_fn)
        logging.info('(Eval)  step %6i, loss %.6f, acc=%.4f', step, loss, acc)

      # Saves checkpoint.
      if step % FLAGS.checkpoint_iter == 0 or step == FLAGS.train_steps - 1:
        ckpt_path = ckpt_manager.save(step)
        logging.info('Saved checkpoint to %s', ckpt_path)
  else:
    # Testing model
    loss, acc, f1, auc  = get_epoch_result(model, dataset_test, loss_fn,
                                           return_f1_auc=True)
    logging.info('(Test)  step %6i, loss %.6f, acc=%.4f, f1=%.4f, auc=%.4f',
                 init_step, loss, acc, f1, auc)
    # Save the result
    result_fpath = os.path.join(FLAGS.output_dir,
                                f'result_{FLAGS.dataset}.json')
    logging.info('Saving result to %s', result_fpath)
    with open(result_fpath, 'w') as fd:
        result = {
            'loss': float(loss),
            'acc': float(acc),
            'f1': float(f1),
            'auc': float(auc)
        }
        json.dump(result, fd, indent=4)


if __name__ == '__main__':
  app.run(main)
