# Adaptive-WF-Attack

The implementation code of Proteus. Proteus dynamically adapts the model to accommodate shifts in network traffic patterns, improving the resilience of website fingerprinting techniques. 

## Usage

The **Proteus** is built upon the WFlib library. We need clone the [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) repository.

### Install
```
mv Website-Fingerprinting-Library wflib
cd wflib
pip install --user .
```

### Dataset Split
```
python wflib/exp/dataset_process/dataset_split.py --dataset CW
```

### Run Proteus
```
bash proteus.sh
```