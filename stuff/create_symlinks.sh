VAR=/var/job/c5e46d/scratch
mkdir -p work_dir/cifar10
mkdir -p work_dir/church
mkdir -p work_dir/bedroom
mkdir -p work_dir/imagenet
mkdir -p work_dir/cats/base
mkdir -p work_dir/cats/upsampler
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/cifar10/checkpoint_8.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/cifar10/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/cifar10/genie_checkpoint_20000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/cifar10/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/church/checkpoint_300000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/church/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/church/genie_checkpoint_35000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/church/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/bedroom/checkpoint_300000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/bedroom/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/bedroom/genie_checkpoint_40000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/bedroom/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/imagenet/checkpoint_400000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/imagenet/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/imagenet/genie_checkpoint_25000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/imagenet/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/imagenet/cond_checkpoint_400000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/imagenet/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/imagenet/cond_genie_checkpoint_15000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/imagenet/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/cats/base/checkpoint_400000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/cats/base/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/cats/base/genie_checkpoint_20000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/cats/base/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/cats/upsampler/checkpoint_150000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/cats/upsampler/
ln -s $VAR/ddn/checkpoints/genie_checkpoints/higher_score_flow_code/work_dir/cats/upsampler/genie_checkpoint_20000.pth $VAR/zfs/higher_score_flow_code_sync/work_dir/cats/upsampler/
