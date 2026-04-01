# Data Directory

Place the extracted benchmark dataset folder directly in this `data/` directory.

## Expected location

After downloading and extracting the dataset, the structure should look like this:

```text
SIComp/
├── data/
│   ├── README.md
│   └── SIComp_Benchmark/
│       ├── SetA_Surrogate/
│       ├── SetA_Real_Compensation/
│       └── SetB_Real_Compensation/
```

## Important

The dataset folder name should be:

```text
SIComp_Benchmark
```

So the expected benchmark paths are:

```text
data/SIComp_Benchmark/SetA_Surrogate
data/SIComp_Benchmark/SetA_Real_Compensation
data/SIComp_Benchmark/SetB_Real_Compensation
```

## Current script usage

The current reproduction script mainly uses:

```text
data/SIComp_Benchmark/SetA_Surrogate
data/SIComp_Benchmark/SetA_Real_Compensation
```

`SetB_Real_Compensation` is also part of the released benchmark package.

## Common mistake

Do not place the dataset one level deeper like this:

```text
data/SIComp_Benchmark/SIComp_Benchmark/...
```

The folder should be placed directly under `data/`.

For dataset download information and project usage instructions, please see the top-level `README.md` in this repository.
