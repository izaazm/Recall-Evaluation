#!/bin/sh
docker run --name cir-container --gpus all -it --rm -v $(pwd):/app cir-pytorch