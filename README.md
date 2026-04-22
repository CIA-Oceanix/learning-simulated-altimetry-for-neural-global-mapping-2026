# End-to-end learning from simulated observations for the neural global-scale mapping of real altimetry data

To date, this repository contains the data & tools used to generate the figures
from the article.

**Coming soon**: the code for training & inference.


## Installation

1. Clone this repository;
2. Install the environment:
    ```sh
    conda create -n <name_of_your_environment>
    conda activate <name_of_your_environment>
    conda env update -f environment.yaml
    ```


## Figures

To generate the figures used in the article,

1. Download the sources data (used to generate the figures):
    ```
    cd figures
    mkdir resources
    cd resources

    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/stat_sla_fdvarnet_unet.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/stat_sla_miost.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/stat_sla_neurostssh.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/stat_sla_unet_noensemble.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/stat_sla_unet.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/scores_unet.csv
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/psd_sla_fdvarnet_unet.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/psd_sla_miost.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/psd_sla_neurostssh.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/psd_sla_unet_noensemble.nc
    wget https://s3.eu-central-1.wasabisys.com/melody/global-mapping-ose/figures/resources/psd_sla_unet.nc
    ```
2. Run the notebooks located in the directory `figures`.

The notebooks will output the figures and a copy of these figures will
be stored at `figures/images`.
