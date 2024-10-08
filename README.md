# MTL_Toxicity_Constructive

## Create docker container

```bash
docker run --name MTL_TC -it --net=host --gpus '"device=7"' -v /raid/gplsi/robiert/docker_vol/MTL_Toxicity_Constructive/:/workspace -v /raid/gplsi/NAS/GPLSI/:/workspace/NAS nvcr.io/nvidia/pytorch:24.02-py3 bash
```

## Run training script

```bash
PYTHONPATH=src python src/scripts/train_predict_model_mult.py
```