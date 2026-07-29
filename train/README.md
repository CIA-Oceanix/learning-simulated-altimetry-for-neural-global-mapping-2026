# Training code

## Steps

1. Install the conda environment (see the root README)
2. Download the datasets and specify their locations in the configuration
    file `config/xp/base.yaml`:
    ```yaml
    datamodule:
      _target_: ...
      input_da:
        tgt_path: ???  # location to the ground truth file
        ...
        inp_path: ???  # location to the input file
        ...
    ```
3. Run the models :)

## Train a model

To run the **UNet** model, execute the following command:

```
python train.py xp=unet
```

Use `xp=fdvarnet-convlstm` if you would like to run the **4DVarNet-ConvLSTM**
model instead.


## Fine-tune a model

If you want to fine-tune a pre-trained model, you can specify the path to the
checkpoint in parameter `finetune`:

```
python train.py xp=unet finetune=outputs/my_model/checkpoints/best.ckpt
```