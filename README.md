# ResNET on pico in C

Architecture : https://chazamfr.github.io/resnet-pico-web/

```bash
git clone https://github.com/ChaZamFr/ResNET-PICO.git
cd ResNET-PICO
```
```bash
git clone https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk
git submodule update --init
cd ..
```
```bash
python resnet_train.py
```
resnet_train.py gives the weights of the model in the form of header file for our C code as weights_mnist.h

```bash
mkdir build
cd build
cmake ..
make
```

Then we can flash the .elf in our PICO and the model is running on PICO
To check if the model is running or not :

```bash
ls /dev/ttyACM*
screen /dev/ttyACM*
```
Check the ttyACM* number then enter the number with screen command then type "start" after the console opens then we can run the model



