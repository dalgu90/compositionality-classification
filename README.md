# CFQ classification task

This repository converts CFQ dataset into a sentence pair classification task and trains deep learning models on the task annotated with structural annotation.

## 1. Environment Setting
The code runs with Tensorflow 2.5. Set up a conda environment as follows.
```shell
# You can name the environment as you like.
$ conda create --name cfq python=3.8 cudatoolkit=11.3 cudnn=8.2
$ conda activate cfq
$ pip install tensorflow==2.5 numpy==1.19.5 scikit-learn==0.23.1
```

## 2. Download the dataset
Download the dataset file from [here](https://drive.google.com/file/d/1dcZW7Z66GwtH3wBRehRBvIxWZmWmVpwx/view?usp=drive_link) and extract it. Place the resulting `data` directory at the project root.
- Please refer to dataset.md for details on how to create the dataset.
- **Note**: As of now, only the dataset without structure annotations is available.

<!---
**Note**: Currently, dataset with structure annotation (or when `output_tree` is `true`) can be generated when `xlink_mapping.pkl` is placed under the dataset output dir (This file can be generated using a jupyter notebook `colab/cfq_xlink_mutual_information.ipynb` and the dataset of the same config but without structure annotation).
-->

## 3. Run model
Run one of the training shell scripts (`cls_*.sh`) in the project root (Please see help output for usage). Note that only Relative Transformer can use structure annotations. For example,
```shell
# Train and test an LSTM on the Random Split & Random Neg dataset
$ bash cls_lstm.sh random_random
# Train and test a Transformer (6 layers) on the MCD Split & Model Neg dataset
$ bash cls_relative_transformer_nomask.sh mcd_model 6
```

## Dependencies
Checkout the ETC repository at (https://github.com/google-research/google-research/tree/master/etcmodel) under the `third_party` directory.

## Cite This Work
```
@inproceedings{kim2021improving,
  title={Improving Compositional Generalization in Classification Tasks via Structure Annotations},
  author={Kim, Juyong and Ravikumar, Pradeep and Ainslie, Joshua and Onta{\~n}{\'o}n, Santiago},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
  pages={637--645},
  year={2021},
}
```

### TODO List
- [x]  Create splits for model negative (for `{random|mcd}_model`)
- [x]  Train LSTM/Transformer on the CFQ seq-to-seq task (natural language question -> SPARQL query)
- [x]  Generate decoding results from CFQ seq-to-seq models
- [x]  Generate the CFQ sentence pair classification dataset w/o structure annotations (`{random|mcd}_{random|model}`)
- [x]  Run classification experiments on the datasets above with LSTM/Transformer
- [ ]  Parse questions and queries to generate masks
- [ ]  Generate the CFQ classification dataset with structure annotations (`mcd_model_tree`)
- [ ]  Run classification experiments on the dataset above with Relative Transformer
