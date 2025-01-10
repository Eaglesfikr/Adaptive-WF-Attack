# Adaptive-WF-Attack

The implementation code of Proteus. Proteus dynamically adapts the model to accommodate shifts in network traffic patterns, improving the resilience of website fingerprinting techniques. 

## Usage

### Install
```
cd wflib_copy
pip install --user .
cd ..
```

### Dataset Split
```
python exp/dataset_process/dataset_split.py --infile datasets/TimeDrift/240313.npz
```

### Run Proteus
```
bash scripts/TimeDrift/Proteus.sh
```