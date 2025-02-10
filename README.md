# PTCNet
PyTorch implementation of Paper "PTCNet: Pure Transformer Network for Two-View Correspondence Pruning"

Part of the code is borrowed or ported from [OANet](https://github.com/zjhthu/OANet) and [CLNet](https://github.com/sailor-z/CLNet). Please also cite these works if you find the corresponding code useful.

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.

## Datasets

Download the YFCC100M dataset and the SUN3D dataset from the [OANet](https://github.com/zjhthu/OANet) repository.

# Testing and Training Model

The results in our paper can be reproduced by running the test script:
```bash
cd core 
python main.py --run_mode=test --model_path=../model-yfcc-sift/
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

If you want to retrain the model on YFCC100M, run the tranining script.
```bash
cd core 
python main.py 
```

You can also retrain the model on SUN3D by modifying related settings in `code\config.py`.
