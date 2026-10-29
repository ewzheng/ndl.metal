# ndl.metal(): A Needle Backend for Apple GPUs and Neural Accelerators
## F26 10-714: Deep Learning Systems Final Project

Needle (Necessary Elements of Deep Learning) is a simple machine learning framework for Python. The standard implementation in 10-714 provides backends for CPUs and NVIDIA GPUs (via CUDA), which have been the gold standard for deep learning since the AI boom. Apple silicon devices are common in the wild and increasingly used for deep learning, but Needle does not support it. Our implementation aims to bridge this gap with a Metal backend for Apple silicon-based SoCs, and to extend it to the Neural Accelerators in the latest M5 chips.

---
### Team

**Yujue Wang**
**Yi (Shirley) Xu**
**Elias Zheng**

---

## Setup on MacOS:
1. Install XCode Command Line Tools:
```
xcode-select --install
``` 
2. Install CMake: (ensure brew is installed beforehand)
```sh
brew install cmake
```
3. Create a Python virtual environment and install dependencies:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install pybind11 numpy pytest torch
pip install git+https://github.com/dlsys10714/mugrade.git
```
4. Build project:
```sh
make
```

