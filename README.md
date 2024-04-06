# Object detection on edge devices
# Arduino Nano 33 BLE Sense
### Connect board to Edge Impulse
- Download the [Edge Impulse firmware](https://cdn.edgeimpulse.com/firmware/arduino-nano-33-ble-sense.zip) 
- Flash the firmware
- Run ```edge-impulse-daemon```

### Using model
After training model on Edge Impulse, download binary containing both the Edge Impulse data acquisition client and full impulse. Flash the firmware onto the board and execute the following command:
```
edge-impulse-run-impulse --debug
```
Real-time object detection can be observed via web browser.

# Raspberry Pi 4
### Installing dependencies for Edge Impulse
```
sudo apt update
curl -sL https://deb.nodesource.com/setup_16.x | sudo bash -
sudo apt install -y gcc g++ make build-essential nodejs sox gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps
npm config set user root && sudo npm install edge-impulse-linux -g --unsafe-perm
```

## 1. Image object detection
### Requirements:
- GNU Make
- LLVM 9

### Build and run object detection program
```
cd ./image-object-detection
make -j 4
./build/app test_sample.jpg
```

## 2. Video object detection
Arduino Nano 33 BLE Sense captures frames using an attached camera module and transmitting them to the Raspberry Pi. Upload the ```./arduino/rgb565.ino``` sketch onto the Arduino board to enable this transmission.

### RPi Requirements:
- Python 3 (>= 3.7)

### Install packages
```
pip3 install -r requirements.txt
```

### Install Edge Impulse Python SDK
```
sudo apt-get install libatlas-base-dev libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev
pip3 install edge_impulse_linux -i https://pypi.python.org/simple
```
### Download model file
```
edge-impulse-linux-runner --download modelfile.eim
chmod +x modelfile.eim
```

### Run Edge Impulse model
```
python3 inference.py
```

### Run models from TF Hub
|  | Model path | Image size | Tensor Type |
|----------|----------|----------|--------|
| SSD MobileNetV1   | ./hub_models/ssdmobilenetv1.tflite  |  300x300  | uint8 | 
| Efficient Det 1   | ./hub_models/lite1-detection-default.tflite   | 320x320   | uint8 |
| Efficient Det 2   | ./hub_models/lite2-detection-default.tflite   | 448x448   | uint8 |
| YOLOv5   | ./hub_models/yolov5.tflite   | 320x320   | float32 |

```
python3 inference_tflite.py
```
Since YOLOv5 has different output format, run:
```
python3 inference_yolotflite.py
```

### Run models created with TF Model Maker
|  | Model path | Image size | Tensor Type |
|----------|----------|----------|--------|
| Efficient Det 0  | ./colab_models/model-effi_0.tflite   | 300x300   | uint8 |
| Efficient Det 2   | ./hub_models/model-effi_2.tflite   | 448x448   | uint8 |
```
python3 inference_colab_tflite.py
```
