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

## Conda commands

Export environment:

```
conda env export --no-builds | grep -v "prefix" > environment.yml
```

Upgrade packages:

```
conda update --all
```