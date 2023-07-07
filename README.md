# CL-VIME
Implemented [VIME](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf) with Contrastive Learning added with pytorch


## Code explanation
1. conf/config.yaml
    - File for manegement hyperparameters
2. data_loader.py
    - Load and preprocess MNIST and other tabular data
3. [infonce.py](https://github.com/RElbers/info-nce-pytorch)
    - Contrastive loss function
4. main.py
    - Executable file (adjusting hyperparameters using hydra)
5. model.py
    - Models required for VIME and Contrastive Learning
6. train.py
    - Training and test models
7. utils.py
    - Some utility functions for metrics and contrastive learning, VIME frameworks


## Requirement
- Python = 3.8.10

```bash
python -m venv venv
. venv/bin/activate
```

## Installation
```bash
pip install -r requirements.txt
```

## Command
```bash
python main.py
```
