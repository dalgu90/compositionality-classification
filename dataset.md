# Generation of CFQ Classification Dataset

This note describes the steps to regenerate our CFQ sentence pair classification dataset. To obtain the decoding results from the CFQ baseline models, we use an earlier commit of the `google/google-research` repository and Tensorflow 1.15. This setup requires CUDA GPUs from the RTX 20XX series or earlier models.

## 1. CFQ repository setup
1. Create a conda environment.
```shell
# You can name the environment as you like.
$ conda create --name cfq_baseline cudatoolkit=10.0 numpy=1.19 tensorflow-gpu=1.15
$ conda activate cfq_baseline
$ pip install setuptools==59.5.0
$ pip install gym==0.18.3 dopamine-rl==3.0.1 tensorflow-datasets==3.2.1 tensor2tensor==1.15.7 ijson==3.1
```
2. Clone the Google Research repository into a *separate* directory, and checkout to an old commit.
```shell
# On a separate directory
$ git clone git@github.com:google-research/google-research.git
$ cd google-research/cfq
$ git checkout a8e876be
```
3. Download the CFQ dataset (https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz) and extract.
```shell
# Place the CFQ dataset file under google-research/cfq and extract.
$ tar xzvf cfq1.1.tar.gz --strip-components=1
```
4. Create a symlink of the CFQ directory under `scripts` directory of our repo.
```shell
# A `cfq` symlink should appear under `scripts` after this.
$ ln -s `pwd` (path-to-our-repo)/scripts
```

## 2. Generate CFQ split and copy files
1. Generate CFQ splits. In the `scripts` directory of our repo,
```shell
$ python convert_cfq_splits.py --split_file random.json
$ python convert_cfq_splits.py --split_file mcd.json
$ cp cfq_splits/*.json cfq/splits/
```
2. Copy the CFQ baseline training script
```shell
$ cp run_cfq_baselines.sh cfq/
```

## 3. Run CFQ baseline models for model negatives
1. Run the CFQ baseline and get decoding results. In the `scripts` directory of our repo,
```shell
$ cd cfq  # Navigate to the symlink
# This may takes some time
$ for cfq_split in {random,mcd}; do for model in {lstm,transformer,universal}; do bash run_cfq_baseline.sh $cfq_split $model; done; done
```
After running it, the decoding results with the dataset should look like this:
```
scripts/cfq -> (path-to-google-research)/cfq
└─cfq_model_outputs
  ├─mcd
  │ ├─train_encode.txt
  │ ├─train_decode.txt
  │ ├─train_decode_lstm.txt
  │ ├─train_decode_transformer.txt
  │ ├─train_decode_universal.txt
  │ ├─dev_encode.txt
  │ ...
  │ ├─dev_decode_universal.txt
  │ ├─test_encode.txt
  │ ...
  │ └─test_decode_universal.txt
  └─random
    ├─train_encode.txt
    ...
    └─test_decode_universal.txt
```

## 4. Create CFQ classification dataset (w/o structure annotations) using CFQ model outputs
2. Run the generation script. In the `scripts` directory of our repo,
```shell
$ for cfq_split in {random,mcd}; do for neg_method in {random,model}; do bash create_cls_dataset.sh $cfq_split $neg_method true notree; done; done
```

## 5. Create CFQ classifictaion dataset (w. structure annotations)
TODO
