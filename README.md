# Enhancing Website Fingerprinting Attacks against Traffic Drift


This paper introduces Proteus, an adaptive framework that enhances the robustness of website fingerprinting (WF) attacks against real-world traffic drift by fine-tuning models with unlabeled traffic.


## Datasets

From March to December 2024, we collected over 300 thousand real-world Tor traffic under various traffic drift scenarios. 

Our dataset includes six distinct categories of data, which can be downloaded via the provided [link](https://drive.google.com/drive/folders/1itVf6TYjGWvGrVQQsxLdnXk7E_AVTL1m).


```sh
mkdir datasets
```

Download and extract all datasets, then move them to the `datasets` folder.

## Usage

### Install

```sh
cd wflib_copy
pip install --user .
cd ..
```

### Experiments

- Section 5.2
```
bash scripts/TemporalDrift/Proteus.sh
```

- Section 5.3
```
bash scripts/VersionDrift/Proteus.sh
```

- Section 5.4
```
scripts/NetworkDrift/Proteus.sh
```

- Section 5.5
```
scripts/BehaviorDrift/Proteus.sh
```

- Section 5.6
```
bash scripts/OpenWorld/Proteus.sh
```

- Section 5.7
```
bash scripts/Defense/Proteus.sh
```