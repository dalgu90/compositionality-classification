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
"""Function to convert the CFQ splits into splits for model negatives. """

import json
import os
import random

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('split_root', 'cfq_splits', 'Path to cfq split dir')
flags.DEFINE_string('split_file', None, 'The MCD split file')
flags.DEFINE_integer('seed', 123, 'Random seed')

flags.register_validator('split_root', os.path.exists, 'CFQ split not found.')


def create_model_neg_splits(orig_split):
  """Create splits for model neg splits"""
  train_idxs = orig_split['trainIdxs']
  random.shuffle(train_idxs)
  trainA_idxs = train_idxs[:len(train_idxs) // 2]
  trainB_idxs = train_idxs[len(train_idxs) // 2:]
  dev_idxs = orig_split['devIdxs']
  test_idxs = orig_split['testIdxs']

  # Test split is not important here, but should not be empty.
  model_neg_splits = {
    'trainA': {'trainIdxs': trainA_idxs, 'devIdxs': trainB_idxs, 'testIdxs': test_idxs},
    'trainB': {'trainIdxs': trainB_idxs, 'devIdxs': trainA_idxs, 'testIdxs': test_idxs},
    'dev': {'trainIdxs': dev_idxs, 'devIdxs': test_idxs, 'testIdxs': trainA_idxs},
    'test': {'trainIdxs': test_idxs, 'devIdxs': dev_idxs, 'testIdxs': trainA_idxs},
  }

  return model_neg_splits


def main(argv):
  del argv  # unused

  # Use the random seed for reproducibility
  random.seed(FLAGS.seed)

  # Load the original CFQ split
  orig_split_path = os.path.join(FLAGS.split_root, FLAGS.split_file)
  logging.info(f'Load CFQ split: {orig_split_path}')
  with open(orig_split_path) as fd:
      orig_split = json.load(fd)

  # Create splits for model negative training
  orig_split_name = os.path.splitext(FLAGS.split_file)[0]
  model_neg_splits = create_model_neg_splits(orig_split)
  for new_name, new_split in model_neg_splits.items():
    new_split_path = os.path.join(FLAGS.split_root,
                                f'{orig_split_name}_{new_name}.json')
    logging.info(f'Save CFQ split: {new_split_path}')
    with open(new_split_path, 'w') as fd:
      json.dump(new_split, fd)

  logging.info('Done!')


if __name__ == "__main__":
  app.run(main)
