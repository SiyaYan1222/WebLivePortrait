#!/bin/bash

# warping+spade model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx_v1.1/warping_spade-fix-v1.1.onnx
# motion_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx_v1.1/motion_extractor-v1.1.onnx -p fp32
# appearance_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx_v1.1/appearance_feature_extractor-v1.1.onnx
# stitching model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx_v1.1/stitching-v1.1.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx_v1.1/stitching_eye-v1.1.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx_v1.1/stitching_lip-v1.1.onnx

mkdir -p ./checkpoints/liveportrait_animal_onnx
mv ./checkpoints/liveportrait_animal_onnx_v1.1/*.trt ./checkpoints/liveportrait_animal_onnx/