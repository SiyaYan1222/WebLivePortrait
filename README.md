# WebLivePortrait: Web Live Portraits in Real Time

## Background Context
Live Portrait is a `deep learning–based model` for `face reenactment` and `motion transfer`.
It takes a `source image (or video)` and a `driving video`, and generates an `animated output` where the source face follows the expressions and movements of the driver.

#### 🧩 Mechanism

- Motion Extraction – Extracts motion keypoints (e.g., head pose, facial expressions) from the driving video.

- Appearance Feature Extraction – Encodes visual features from the source image/video.

- Warping & Stitching – Warps the source features based on motion, then stitches them into a coherent animated frame.

## Introduction
This repository is a fork of [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait), which itself is based on the original [LivePortrait](https://github.com/KwaiVGI/LivePortrait) project.

In this project, I focused on retaining only the core functionality I needed from FasterLivePortrait and extending it with new features.

#### ✅ Features kept from FasterLivePortrait

- Source image/video + driving video → Live Portrait (CLI & WebUI)

- Webcam real-time input → Live Portrait (CLI)

- OnnxGPU and TensorRT option for GPU usage (Utilize TensorRT for Real Time Usage)

#### ✨ New feature added in this fork

- Webcam real-time generation in Gradio Web App for smoother, interactive usage.

## Environment Setup
  * Install [Docker](https://docs.docker.com/desktop/install/windows-install/) according to your system
  * Download the image: `docker pull shaoguo/faster_liveportrait:v3`
  * Clone this [git repository](https://github.com/SiyaYan1222/WebLivePortrait.git), Go to the WebLivePortrait folder
  * Download all ONNX and TRRT model files:`huggingface-cli download ssyyaa/liveportrait --local-dir ./checkpoints`
    * (Optional) Download Just ONNX model files:`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`
    * (If TensorRT availiable) Manully convert all ONNX models to TensorRT, run `sh scripts/all_onnx2trt.sh` and `sh scripts/all_onnx2trt_animal.sh`


## Usage

### Script to Run
  * Start Docker Container
    ```sh
    ./runDocker.sh start
    ```
  * Start WebUI mode ( `http://localhost:9870/`)
    ```sh
    ./runDocker.sh webui
    ```
  * Start Webcam Real-Time mode in terminal
    ```sh
    ./runDocker.sh webcam
    ```
  * Go to the docker container shell
    ```sh
    ./runDocker.sh shell
    ```
  * Stop the Docker Container
    ```sh
    ./runDocker.sh stop
    ```
  * Remvoe the Docker Container
    ```sh
    ./runDocker.sh remove
    ```

### Start Docker Container Shell Manully
  ```shell
      docker run -it --gpus=all --name web_liveportrait  \
      --device /dev/video0 --device /dev/video1 \
      --device /dev/nvidia0 --device /dev/nvidiactl \
      --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
      -v /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:ro \
      -v $(pwd -P):/root/WebLivePortrait \
      -p 9870:9870 shaoguo/faster_liveportrait:v3 /bin/bash
  ```

### Use CLI in Docker Container Shell

### Install Python Dependency libs:
  ```shell
   cd root/WebLivePortrait 
   pip install -r requirements.txt
  ```
### Test the basic pipeline:
  ```shell
   python run.py \
   --src_image assets/examples/source/s3.jpg \
   --dri_video assets/examples/driving/d3.mp4 \
   --cfg configs/trt_infer.yaml # --cfg configs/onnx_infer.yaml
  ```
### Run real-time with webcam:
  ```shell
   python run.py \
   --src_image assets/examples/source/s3.jpg \
   --dri_video 0 \
   --cfg configs/trt_infer.yaml --realtime # --cfg configs/onnx_infer.yaml
  ```
### Run all in Gradio WebUI:
  ```shell
   python webui.py # python webui.py --mode onnx
  ```
  * The default port is 9870. Open the webpage: `http://localhost:9870/`



## License

- **Code**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- **Models**: Any machine learning models used in this project are subject to their respective licenses. Please refer to the original model sources for license information. We do not take responsibility for model license compliance.


