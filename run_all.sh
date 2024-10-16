# python train.py -s data/nerf_synthetic/chair -m outputs/test --gui --densify_grad_threshold 0.0002 --eval
# python train.py -s data/nerf_synthetic/chair -m outputs/nerf/chair --eval
# python train.py -s data/nerf_synthetic/drums -m outputs/nerf/drums --eval
# python train.py -s data/nerf_synthetic/ficus -m outputs/nerf/ficus --eval
python train.py -s data/nerf_synthetic/hotdog -m outputs/nerf/hotdog --eval
python train.py -s data/nerf_synthetic/lego -m outputs/nerf/lego --eval
python train.py -s data/nerf_synthetic/materials -m outputs/nerf/materials --eval
python train.py -s data/nerf_synthetic/mic -m outputs/nerf/mic --eval
python train.py -s data/nerf_synthetic/ship -m outputs/nerf/ship --eval