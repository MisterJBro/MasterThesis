# MasterThesis

## Installation
Create an conda environment with all dependencies via:

```
conda env create -f environment.yml
conda activate rl

```

Then install the hex rust environment via:

```
cd gym/hex
maturin develop --release
```

## Execution
Add module to path, like
```
set PYTHONPATH=D:/Documents/TU Darmstadt/Master/4 Semester/Master Thesis/MasterThesis/
```

## Conda commands

Export environment:

```
conda env export --no-builds | grep -v "prefix" > environment.yml
```

Upgrade packages:

```
conda update --all
```

```
tensorboard --logdir experiments/ --samples_per_plugin images=200
```
