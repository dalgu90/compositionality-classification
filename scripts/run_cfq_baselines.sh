#!/bin/bash

# Copyright 2020 The Google Research Authors.
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

set -e
set -x

#virtualenv -p python3 .
#source ./bin/activate
#pip3 install -r cfq/requirements.txt

# Run CFQ baselines for the random/equal MCD splits
# You need to run the conversion script (convert_cfq_splits.py),
# and put `${cfq_split}_{trainA,trainB,dev,test}.json` under `splits`

# CFQ split. Choose one of these
cfq_split=$1
#cfq_split="random"
#cfq_split="mcd"

# CFQ baseline model. Choose one of these
model=$2
#model="lstm"
#model="transformer"
#model="universal"

# Preset for baseline models
if [[ "$model" -eq "lstm" ]]; then
    # LSTM:
    cfq_model="lstm_seq2seq_attention"
    hparams_set="cfq_lstm_attention_multi"
    train_steps="35000"
elif [[ "$model" -eq "transformer" ]]; then
    # Transformer:
    cfq_model="transformer"
    hparams_set="cfq_transformer"
    train_steps="35000"
elif [[ "$model" -eq "universal" ]]; then
    # Universal Transformer:
    cfq_model="universal_transformer."
    hparams_set="cfq_universal_transformer"
    train_steps="35000"
fi

# Train & decode
#for i in $3; do
for i in 1 2 3 4; do
    if [[ "$i" -eq "1" ]] ; then
        neg_train="trainA"
        neg_dev="trainB"
    elif [[ "$i" -eq "2" ]] ; then
        neg_train="trainB"
        neg_dev="trainA"
    elif [[ "$i" -eq "3" ]] ; then
        neg_train="dev"
        neg_dev="test"
    elif [[ "$i" -eq "4" ]] ; then
        neg_train="test"
        neg_dev="dev"
    fi

    data_name="${cfq_split}_${neg_train}"
    save_path="t2t_data_${data_name}"
    neg_train_path="${save_path}/train/train_encode.txt"

    if [ ! -f "$neg_train_path" ] ; then
        # 1) Generate dataset split
        python3 -m preprocess_main --dataset_path="dataset.json" \
            --split_path="splits/${data_name}.json" \
            --save_path="${save_path}"

        # 2) Create TF record files
        t2t-datagen --t2t_usr_dir="$(pwd)/cfq/" \
            --data_dir="${save_path}" \
            --problem="cfq" \
            --tmp_dir="/tmp/cfq_tmp_$RANDOM"
    fi

    # 3) Train a CFQ baseline
    if [ ! -e "${save_path}/output_${model}/model.ckpt-${train_steps}.meta" ]; then
      t2t-trainer --t2t_usr_dir="$(pwd)/cfq/" \
          --data_dir="${save_path}" \
          --problem="cfq" \
          --model="${cfq_model}" \
          --hparams_set="${hparams_set}" \
          --output_dir="${save_path}/output_${model}" \
          --train_steps="${train_steps}"
    fi

    # 4) Decode with the trained model
    if [ ! -d "cfq_model_outputs/${cfq_split}" ]; then
        mkdir -p cfq_model_outputs/${cfq_split}
    fi

    if [ ! -e "cfq_model_outputs/${cfq_split}/${neg_dev}_encode.txt" ]; then
      cp ${save_path}/dev/dev_encode.txt cfq_model_outputs/${cfq_split}/${neg_dev}_encode.txt
      cp ${save_path}/dev/dev_decode.txt cfq_model_outputs/${cfq_split}/${neg_dev}_decode.txt
    fi

    t2t-decoder --t2t_usr_dir="$(pwd)/cfq/" \
        --data_dir="${save_path}" \
        --problem="cfq" \
        --model="${cfq_model}" \
        --hparams_set="${hparams_set}" \
        --output_dir="${save_path}/output_${model}" \
        --checkpoint_path="${save_path}/output_${model}/model.ckpt-${train_steps}" \
        --decode_from_file="${save_path}/dev/dev_encode.txt" \
        --decode_to_file="cfq_model_outputs/${cfq_split}/${neg_dev}_decode_${model}.txt" \
        --decode_hparams="return_beams=True,write_beam_scores=True"
done
