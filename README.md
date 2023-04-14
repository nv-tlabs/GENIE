# <p align="center">GENIE: Higher-Order Denoising Diffusion Solvers <br><br> NeurIPS 2022</p>

<div align="center">
  <a href="https://timudk.github.io/" target="_blank">Tim&nbsp;Dockhorn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://latentspace.cc/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://karstenkreis.github.io/" target="_blank">Karsten&nbsp;Kreis</a>
  <br> <br>
  <a href="https://arxiv.org/abs/2210.05475" target="_blank">Paper</a> &emsp;
  <a href="https://nv-tlabs.github.io/GENIE/" target="_blank">Project&nbsp;Page</a> 
</div>
<br><br>
<p align="center">
    <img width="750" alt="Animation" src="assets/genie_pipeline.png"/>
</p>

## Requirements

GENIE is built using PyTorch 1.11.0 and CUDA 11.3. Please use the following command to install the requirements:
```shell script
pip install -r requirements.txt 
``` 
Optionally, you may also install [NVIDIA Apex](https://github.com/NVIDIA/apex). The Adam optimizer from this library is faster than PyTorch's native Adam.

## Pretrained checkpoints

We provide [pre-trained checkpoints](https://drive.google.com/drive/folders/18BBkidk0pSs1skYSKVH86pJNcsJHJhrU?usp=sharing) for all models presented in the paper. Note that the CIFAR-10 base diffusion model is taken from the [ScoreSDE repo](https://github.com/yang-song/score_sde_pytorch).

| Description | Checkpoint path |
|:----------|:----------|
| CIFAR-10 base diffusion model | [`work_dir/cifar10/checkpoint_8.pth`](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view?usp=share_link) |
| CIFAR-10 base GENIE model | [`work_dir/cifar10/genie_checkpoint_20000.pth`](https://drive.google.com/file/d/1MUYpgHoqo7zrS93a-IH0UWU3j3FelWny/view?usp=share_link) |
| Church base diffusion model | [`work_dir/church/checkpoint_300000.pth`](https://drive.google.com/file/d/1s1sDyfep6nd7b8D5E85DHGSkHwW_oAiA/view?usp=share_link) |
| Church base GENIE model | [`work_dir/church/genie_checkpoint_35000.pth`](https://drive.google.com/file/d/1lbXzqkbl0ho9krVySVLj_1bFjdkyXU5r/view?usp=share_link) | 
| Bedroom base diffusion model | [`work_dir/bedroom/checkpoint_300000.pth`](https://drive.google.com/file/d/1xRtkzinz5Ft4PZeTR0Ob2TFOMCt1Xr_-/view?usp=share_link) |
| Bedroom base GENIE model | [`work_dir/bedroom/genie_checkpoint_40000.pth`](https://drive.google.com/file/d/1qgPPLQ6bYKLIh3gr41Ni-GN-8k6RoQvA/view?usp=share_link) | 
| ImageNet base diffusion model | [`work_dir/imagenet/checkpoint_400000.pth`](https://drive.google.com/file/d/1IJa38CLQhyN_e7z1rRL-CMDBJI2FmxJT/view?usp=share_link) |
| ImageNet base GENIE model | [`work_dir/imagenet/genie_checkpoint_25000.pth`](https://drive.google.com/file/d/1KxiugMAJ11JjUadZz4U7esNtLupeqct3/view?usp=share_link) | 
| Conditional ImageNet base diffusion model | [`work_dir/imagenet/cond_checkpoint_400000.pth`](https://drive.google.com/file/d/19wwQckiCtk_KPY9snWdkL996z_5tgybH/view?usp=share_link) |
| Conditional ImageNet base GENIE model | [`work_dir/imagenet/cond_genie_checkpoint_15000.pth`](https://drive.google.com/file/d/1grNa7YbH22EdF3CIF5kz9V50XH1Cuwg_/view?usp=share_link) | 
| Cats base diffusion model | [`work_dir/cats/base/checkpoint_400000.pth`](https://drive.google.com/file/d/1M0Q3JzNESMzNRVnjVHhGB_1qr8HRK55-/view?usp=share_link) |
| Cats base GENIE model | [`work_dir/cats/base/genie_checkpoint_20000.pth`](https://drive.google.com/file/d/1L-iwcGkSv52fIvKbXfwJDrGLQ9Q6-9CV/view?usp=share_link) | 
| Cats diffusion upsampler | [`work_dir/cats/upsampler/checkpoint_150000.pth`](https://drive.google.com/file/d/1IxZ5NvKmGKFNYiJzxQ_1UTv1nxl5Ad3V/view?usp=share_link) |
| Cats GENIE upsampler | [`work_dir/cats/upsampler/genie_checkpoint_20000.pth`](https://drive.google.com/file/d/1z8Lc7QSY9CkwMXwsFczDr0cfDiji5cUr/view?usp=share_link) | 

## Unconditional sampling

After placing the provided checkpoints at the paths outlined above, you can sample from the base model via:

```shell script
python main.py --mode eval --config <dataset>.eval --workdir <new_directory> --sampler ttm2
```

Here, `dataset` is one of `cifar10`, `church`, `bedroom`, `imagenet`, or `cats`. To turn off the GENIE model and sample from the plain diffusion model (via DDIM), simply remove the `--sampler ttm2` flag. By default, the above generates 16 samples using a single GPU.

On the `cats` dataset, we also provide an upsampler, which can be run using the following command:

```shell script
python main.py --mode eval --config cats.eval_upsampler --workdir <new_directory> --data_folder <folder_with_128x128_samples> --sampler ttm2
```

## Conditional and classifier-free guidance sampling

On ImageNet, we also provide a class-conditional checkpoint, which can be controleld via the ``--labels`` flag.

```shell script
python main.py --mode eval --config imagenet.eval_conditional --workdir output/testing_sampling/imagenet_genie_conditional/v2/ --sampler ttm2 --labels 1000
```

To generate all samples from the same class, you can set ``--labels`` to a single integer between 0 and 999 (inclusive). Alternatively, you can provide a list of labels, for example, ``--labels 0,87,626,3``; note, however, that the length of the list needs to be the same as the total number of generated samples. To sample using random labels, you may set the ``--labels`` flag to the number of classes, for ImageNet that would be `1000`.

Furthermore, since we provide both class-conditinal and unconditional checkpoints for ImageNet, you can generate samples using classifier-free guidance:

```shells cript
python main.py --mode eval --config imagenet.eval_with_guidance --workdir output/testing_sampling/imagenet_genie_guidance/v3 --sampler ttm2 --labels 1000 --guidance_scale 1.
```

The ``--guidance_scale`` flag should be set to a positive float.

## Training your own models

### Data preparations

First, create the following two folders:
```shell script
mkdir -p data/raw/
mkdir -p data/processed/
```
Afterwards, run the following commands to download and prepare the data used for training.

<details><summary>CIFAR-10</summary>

```shell script
wget -P data/raw/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
python dataset_tool.py --source data/raw/cifar-10-python.tar.gz --dest data/processed/cifar10.zip
```
</details>
<details><summary>LSUN Chuch</summary>

```shell script
python lsun/download.py -c church_outdoor
mv church_outdoor_train_lmdb.zip data/raw
mv church_outdoor_val_lmdb.zip data/raw
unzip data/raw/church_outdoor_train_lmdb.zip -d data/raw/
python dataset_tool.py --source=data/raw/church_outdoor_train_lmdb/ --dest=data/processed/church.zip --resolution=128x128
```
</details>
<details><summary>LSUN Bedroom</summary>

```shell script
python lsun/download.py -c bedroom
mv bedroom_train_lmdb.zip data/raw
mv bedroom_val_lmdb.zip data/raw
unzip data/raw/bedroom_train_lmdb.zip -d data/raw/
python dataset_tool.py --source=data/raw/bedroom_train_lmdb/ --dest=data/processed/bedroom.zip --resolution=128x128
```
</details>
<details><summary>AFHQ-v2</summary>

```shell script
wget -N https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
mv 'afhq_v2.zip?dl=0' data/raw/afhq_v2.zip
unzip data/raw/afhq_v2.zip -d data/raw/afhq_v2
python dataset_tool.py --source=data/raw/afhq_v2/train/cat --dest data/processed/cats.zip
python dataset_tool.py --source=data/raw/afhq_v2/train/cat --dest data/processed/cats_128.zip --resolution=128x128
```
</details>

<details><summary>ImageNet</summary>

First download the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data), then run the following
```shell script
python dataset_tool.py --source==data/raw/imagenet/ILSVRC/Data/CLS-LOC/train --dest=data/processed/imagenet.zip --resolution=64x64 --transform=center-crop
```
</details>

### FID evaluation

Before training, you should compute FID stats.
```shell script
python compute_fid_statistics.py --path data/processed/cifar10.zip --file cifar10.npz
python compute_fid_statistics.py --path data/processed/church.zip --file church_50k.npz --max_samples 50000
python compute_fid_statistics.py --path data/processed/bedroom.zip --file bedroom_50k.npz --max_samples 50000
python compute_fid_statistics.py --path data/processed/imagenet.zip --file imagenet.npz
python compute_fid_statistics.py --path data/processed/cats.zip --file cats.npz
```

### Diffusion model training scripts

We provide configurations to reproduce our models [here](configs/). Feel free to use a different numbers of GPUs than us, however, in that case, you should also change the (per GPU) batch size (`config.train.batch_size`) in the corresponding config file. To train the base diffusion models, use the following commands:

```shell script
python main.py --mode train --config church.train_diffusion --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config bedroom.train_diffusion --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config imagenet.train_diffusion --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config imagenet.train_diffusion_conditional.py --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config cats.train_diffusion --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config cats.train_diffusion_upsampler --workdir <new_directory> --n_gpus_per_node 8 --n_nodes 2
```

To continue an interrupted training run, you may run the following command:

```shell script
python main.py --mode continue --config <config_file> --workdir <existing_working_directory> --ckpt_path <path_to_checkpoint>
```

We recommend to use the same number of GPUs (via `--n_gpus_per_node`) and nodes (via `--n_nodes`) as in the interrupted run.

### Genie model training scripts

Our GENIE models can be trained using the following commands:

```shell script
python main.py --mode train --config cifar10.train_genie --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config church.train_genie --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config bedroom.train_genie --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config imagenet.train_genie --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config imagenet.train_genie_conditional.py --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config cats.train_genie --workdir <new_directory> --n_gpus_per_node 8
python main.py --mode train --config cats.train_genie_upsampler --workdir <new_directory> --n_gpus_per_node 8 --n_nodes 2
```

To continue interrupted training runs, use the same syntax as above.

## Citation
If you find the provided code or checkpoints useful for your research, please consider citing our NeurIPS paper:

```bib
@inproceedings{dockhorn2022genie,
  title={{{GENIE: Higher-Order Denoising Diffusion Solvers}}},
  author={Dockhorn, Tim and Vahdat, Arash and Kreis, Karsten},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## License

Copyright © 2023, NVIDIA Corporation. All rights reserved.

The code of this work is made available under the NVIDIA Source Code License. Please see our main [LICENSE](./LICENSE) file.

All pre-trained checkpooints are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

#### License Dependencies

For any code dependencies related to StyleGAN3 ([`stylegan3/`](stylegan3/), [`torch_utils/`](torch_utils/), and [`dnnlib/`](dnnlib/)), the license is the  Nvidia Source Code License by NVIDIA Corporation, see [StyleGAN3 LICENSE](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

The [script](./lsun/download.py) to download LSUN data has the MIT License.

We use three diffusion model architectures; see below:
| Model | License |
|:----------|:----------|
| [ScodeSDE](models/score_sde_pytorch) | [Apache License 2.0](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE) |
| [Guided Diffusion](models/guided_diffusion) | [MIT License](https://github.com/openai/guided-diffusion/blob/main/LICENSE) |
| [PyTorch Diffusion](models/pytorch_diffusion) | [MIT License](https://github.com/pesser/pytorch_diffusion/blob/master/LICENSE.md) |