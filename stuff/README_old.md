Getting data:

mkdir -p data/raw/
mkdir -p data/processed/
wget -P data/raw/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
python dataset_tool.py --source data/raw/cifar-10-python.tar.gz --dest data/processed/cifar10.zip

python lsun/download.py -c church_outdoor
python lsun/download.py -c bedroom
mv church_outdoor_train_lmdb.zip data/raw
mv church_outdoor_val_lmdb.zip data/raw
mv bedroom_train_lmdb.zip data/raw
mv bedroom_val_lmdb.zip data/raw
unzip data/raw/church_outdoor_train_lmdb.zip -d data/raw/
unzip data/raw/bedrooms_train_lmdb.zip -d data/raw/

python dataset_tool.py --source=data/raw/church_outdoor_train_lmdb/ --dest=data/processed/church.zip --resolution=128x128
python dataset_tool.py --source=data/raw/bedroomtrain_lmdb/ --dest=data/processed/bedrooms.zip --resolution=128x128

wget -N https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
mv 'afhq_v2.zip?dl=0' data/raw/afhq_v2.zip
unzip data/raw/afhq_v2.zip -d data/raw/afhq_v2
python dataset_tool.py --source=data/raw/afhq_v2/train/cat --dest data/processed/cats.zip

For ImageNet point to https://github.com/CompVis/latent-diffusion codebase

Packages: 
conda create -n genie_code python=3.8.0
pip install -r requirements.txt 

Compute FID stats:
python compute_fid_statistics.py --path data/processed/cifar10.zip --file cifar10.npz
python compute_fid_statistics.py --path data/processed/church.zip --file church_50k.npz --max_samples 50000
python compute_fid_statistics.py --path data/processed/bedrooms.zip --file bedrooms_50k.npz --max_samples 50000

Compute FID stats of generated data:
python compute_fid_statistics.py --path output/church_50k/samples/ --file church_50k_generated.npz --fid_dir output/church_50k/

Compute FID:
python compute_fid.py --path1 assets/stats/church_50k.npz --path2 output/church_50k/church_50k_generated.npz 13.7150

CIFAR-10 GENIE: 
NFE = 10, AFS = 0, Denoising = 0, QS = 0: 7.3577
NFE = 10, AFS = 1, Denoising = 0, QS = 0: 6.9782
NFE = 10, AFS = 0, Denoising = 1, QS = 0: 10.5887
NFE = 10, AFS = 1, Denoising = 1, QS = 0: 9.6617
NFE = 10, AFS = 0, Denoising = 0, QS = 1: 6.0729
NFE = 10, AFS = 1, Denoising = 0, QS = 1: 6.1379
NFE = 10, AFS = 0, Denoising = 1, QS = 1: 6.7714
NFE = 10, AFS = 1, Denoising = 1, QS = 1: 6.1855

Generate cat upsampler:
python scripts/generate_upsampler.py --data_folder output/cats/samples --workdir output/cats_upsampler
python scripts/generate_upsampler.py --data_folder output/cats/samples --workdir output/cats_upsampler --sampler ttm2

Training for now:
python main.py --mode train --config configs/church/train.py --workdir output/training_church/
python main.py --mode train --config configs/cats/train_upsampler.py --workdir output/training_cats_upsampler/

TODO:
[DONE] Fix counter for multi GPU sampling in runners/generate_base.py
[DONE] Test runners/generate_base.py on multiple GPUs
[DONE] Testing FID on church DDIM (python scripts/generate_base.py --dataset church --workdir output/church_50k --batch_size 64 --n_samples 50000)
[DONE] Implement denoising and AFS
[DONE] Upsampler batch
[DONE] Implement diffusion model training
[DONE] Implement upsampler training
Implement GENIE upsampler trianing
Implement GENIE training
Implement classifier-free diffusion model training
Implement classifier-free GENIE training