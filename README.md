# [Source code] End-to-end learning from simulated observations for the neural global-scale mapping of real altimetry data, 2026

This repository contains the training and inference source code used in the
following paper:

> Daniel Zhu, Paul de Nailly, Julien Le Sommer, François Rousseau, Ronan Fablet.
> End-to-end learning from simulated observations for the neural global-scale
> mapping of real altimetry data. 2026.
> ⟨[hal-05598825](https://imt-atlantique.hal.science/hal-05598825v1)⟩.

It also contains the tools used to generate the figures from the paper.


## Installation

1. Clone this repository
2. Install and activate the environment:
    ```sh
    conda create -n <name_of_your_environment>
    conda activate <name_of_your_environment>
    conda env update -f environment.yaml
    ```
3. **Downloads**:
    - For the training and the inference, you will need to download the datasets
        and checkpoints available at the following DOI:
        [10.5281/zenodo.22261913](https://doi.org/10.5281/zenodo.22261913)
    - For plotting the figures, you will need to download the statistics and
        spectral data available at the following DOI:
        [10.5281/zenodo.22644193](https://doi.org/10.5281/zenodo.22644193)

We use [hydra](https://hydra.cc) (included in the environment we provide) to
handle the configurations of our experiments during the training or the
inference.


## Training

1. **Specifying the paths to the datasets**.
    Before performing a training, you must first specify the datasets' paths
    in the configuration file (`./train/config/xp/base.yaml`).

    Let us suppose you have created a directory in `./train/data` in which you
    have put the files:
    - `glorys12_2010_2019_daily_sla_4th_input_gridded_alongtrack.nc`
    - `glorys12_2010_2019_daily_sla_4th_input_mask.nc`
    - `glorys12_2010_2019_daily_sla_4th_target_gridded_alongtrack.nc`

    As described in the paper, there are two training datasets:
    - **A**: the inputs consist in a binary mask (generated from realistic
        altimetry tracks) applied on the Glorys12 ground-truth dataset.
        To enable this dataset, modify the `./train/config/xp/base.yaml` as
        follows:
        ```yaml
        datamodule:
        # remove or comment the following line:
        # _target_: ocean4dvarnet.data.NoisyLazyDataModule

        # use this instead:
        _target_: src.models.MaskingDataModule
        input_da:
            tgt_path: data/glorys12_2010_2019_daily_sla_4th_target_gridded_alongtrack.nc
            inp_path: data/glorys12_2010_2019_daily_sla_4th_input_mask.nc
            inp_var: inp
        ```
    - **B**: the inputs are the Glorys12 ground-truth dataset interpolated
        onto the positions of the 6-nadir constellation flying in 2019, before
        regridding.
        To enable this dataset modify the `./train/config/xp/base.yaml` as
        follows:
        ```yaml
        datamodule:
            input_da:
                tgt_path: data/glorys12_2010_2019_daily_sla_4th_target_gridded_alongtrack.nc
                inp_path: data/glorys12_2010_2019_daily_sla_4th_input_gridded_alongtrack.nc
        ```

    Also, it should be noted that instead of modifying directly the configuration
    file, you could create a new one inheriting `base.yaml` (see
    [hydra](https://hydra.cc)'s documentation), so you can have distinct
    configurations for each dataset.
2. **Training a model**. To start a training, execute the following command in
    your terminal:
    ```sh
    cd train
    python train.py xp=<config>  # `unet` or `fdvarnet-convlstm`
    ```
    A directory `./outputs/yyyy-mm-dd/<xpname>` will be created,
    containing the checkpoints (`.ckpt`) in `<config>/checkpoints` subdirectory
    and the complete configuration used for the training in `.hydra/config.yaml`.
    Both will be needed for the inference.
3. **Resume a training**. If your training has crashed (due to various reason
    like a crash from your GPU or CPU), you can resume the training with:
    ```sh
    cd train
    python train.py xp=<model_you_used> ckpt=<path_to_last_checkpoint>
    ```
4. **Fine-tuning a model**. You have the possibility to fine-tune with:
    ```sh
    cd train
    python train.py xp=<model_you_used> finetune=<path_to_checkpoint>
    ```
    The difference with the resuming is that only weights will be loaded,
    other hyperparameters will not (current epoch, learning rate step, etc).

## Inference

1. **Specifying the paths to the datasets**.
    Before performing an inference, you must first specify the dataset's path
    in the configuration file (`./inference/config/xpi/base.yaml`):
    ```yaml
    observation:
    data:
        _target_: inference.load_dataset
        path: <path_to: nadir-2019-input-6nadir.nc>
    ```
2. **Model**. Let us suppose you already have a trained model in the following
    tree:
    ```sh
    train
    └── outputs
        └── 2026-08-27
            └── unet
                ├── .hydra
                │   └── config.yaml  # complete config file
                └── unet-sla
                    └── checkpoints
                        └── val_mse=1.98889-epoch=425.ckpt  # best model
    ```
    Create a inference configuration file:
    ```
    cd inference/config/xpi
    cp example.yaml unet.yaml
    ```
    Open the newly created `inference/config/xpi/unet.yaml` file and fill the
    fields:
    ```yaml
    method:
        name: unet
        checkpoint: ../train/outputs/2026-08-27/unet/unet-sla/checkpoints/val_mse=1.98889-epoch=425.ckpt
        config: ../train/outputs/2026-08-27/unet/.hydra/config.yaml
    ```
    Then run the following command to perform an inference with the model you
    specified:
    ```sh
    cd inference
    python inference.py xpi=unet
    ```
    At the end of the inference, a reconstruction file will be stored at
    `train/outputs/2026-08-27/unet/unet-sla/checkpoints/`.
3. You can modify the parameter for the ensemble inference by modifying
    `inference/config/xpi/base.yaml` (or by overwriting it):
    ```yaml
    params:
        ensemble: 10  # size of the ensemble
        # ensemble: null  # disable the ensemble inference

    observation:
        noise: 0.2  # standard-deviation of the noise added to each inputs
    ```
4. In the DOI
    [10.5281/zenodo.22261913](https://doi.org/10.5281/zenodo.22261913),
    directory `models`, we share the checkpoints and the YAML config files
    you can reuse to perform inference and reproduce our inferences without
    re-training new models.

    Just specify the paths to the checkpoints and the YAML config in your
    inference configurations.


## Figures

To generate the figures used in the article, run the notebooks located in the
directory `figures` (you will need the data from DOI
[10.5281/zenodo.22644193](https://doi.org/10.5281/zenodo.22644193))

The notebooks will output the figures and a copy of these figures will
be stored at `figures/images`.


## License

This software is a computer program whose purpose is to apply deep learning
schemes to dynamical systems and ocean remote sensing data.

This software is governed by the CeCILL-C license under French law and abiding
by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the terms of the
CeCILL-C license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". As a counterpart to the access to the source code and
rights to copy, modify and redistribute granted by the license, users are
provided only with a limited warranty and the software's author, the holder of
the economic rights, and the successive licensors have only limited liability.

In this respect, the user's attention is drawn to the risks associated with
loading, using, modifying and/or developing or reproducing the software by the
user in light of its specific status of free software, that may mean that it is
complicated to manipulate, and that also therefore means that it is reserved
for developers and experienced professionals having in-depth computer knowledge.
Users are therefore encouraged to load and test the software's suitability as
regards their requirements in conditions enabling the security of their systems
and/or data to be ensured and, more generally, to use and operate it in the same
conditions as regards security. The fact that you are presently reading this
means that you have had knowledge of the CeCILL-C license and that you accept
its terms.
