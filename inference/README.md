# Inference code

## Steps

1. If you have not done it yet, install the Conda environment (see root README)
2. Download the datasets and specify their location in `config/xpi/base.yaml`:
    ```yaml
    observation:
    data:
        _target_: inference.load_dataset
        path: <path_to: sla_filtered_0.25deg.nc>
    ```
