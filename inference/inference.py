import sys
from collections import namedtuple
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import toolz
import torch
import tqdm
import xarray as xr
import xrpatcher
from omegaconf import OmegaConf

sys.path.append('../train')
sys.path.append('/Odyssey/private/d22zhu/Lab/4dvarnet-global-mapping')

torch.set_float32_matmul_precision("high")
PredictItem = namedtuple("PredictItem", ("input",))


class LitModel(pl.LightningModule):
    def __init__(
        self,
        patcher,
        model,
        norm_stats,
        save_dir,
        crop_val,
        out_dims=("time", "lat", "lon"),
        ensemble=None,
        **kwargs,
    ):
        super().__init__()
        self.patcher = patcher
        self.solver = model
        self.norm_stats = norm_stats

        self.ensemble = ensemble

        self.save_dir = Path(save_dir)
        self.save_dir.parent.mkdir(parents=True, exist_ok=True)

        self.out_dims = out_dims
        self.crop_val = crop_val
        self.kwargs = kwargs

    def predict_step(self, batch, batch_idx: int, *args, **kwargs):
        if batch_idx == 0:
            self.predict_data = []

        if self.ensemble:
            outputs = 0.
            for _ in range(self.ensemble["size"]):
                outputs += self.solver(PredictItem(
                    batch.input
                    + torch.randn_like(batch.input) * self.ensemble["noise"],
                ))
            outputs /= self.ensemble["size"]
        else:
            outputs = self.solver(batch)

        batch_size = outputs.shape[0]

        m, s = self.norm_stats
        outputs = outputs.cpu().numpy() * s + m

        num_devices = self.trainer.num_devices * self.trainer.num_nodes
        item_idxes = (
            (batch_idx * batch_size + torch.arange(batch_size))
            * num_devices + self.global_rank
        )

        assert len(self.out_dims) == len(outputs[0].shape)

        for i, idx in enumerate(item_idxes):
            out = outputs[i]
            c = self.patcher[idx].coords.to_dataset()[list(self.out_dims)]
            da = xr.DataArray(out, dims=self.out_dims, coords=c.coords)
            self.predict_data.append(da.astype(np.float32))

    def on_predict_end(self):
        p = self.predict_data[0]
        time, lat, lon = p.time.shape[0], p.lat.shape[0], p.lon.shape[0]

        def _crop(x):
            return crop(x, crop_val=self.crop_val)

        weight = self.kwargs.get(
            "weight",
            build_weight(
                patch_dims={"time": time, "lat": lat, "lon": lon},
                dim_weights={"time": triang, "lat": _crop, "lon": _crop},
            ),
        )
        _cround = self.kwargs.get("_cround", {'lat': 4, 'lon': 4})
        out_coords = self.kwargs["out_coords"]

        ## TODO: actual stuff
        for c, nd in _cround.items():
            out_coords[c] = np.round(out_coords[c], nd)
        out_coords = xr.Dataset(coords=out_coords)
        dims_shape = dict(**out_coords.sizes)

        rec_da = xr.DataArray(
            np.zeros(list(dims_shape.values())),
            dims=list(dims_shape.keys()),
            coords=out_coords.coords,
        )

        count_da = xr.zeros_like(rec_da)
        n_batches = len(self.predict_data)
        for _ in tqdm.tqdm(range(n_batches)):
            da = self.predict_data.pop(0)
            da = da.assign_coords(
                **{c: np.round(da[c].values, nd) for c, nd in _cround.items()}
            )
            w = xr.zeros_like(da) + weight
            wda = da * w
            coords_labels = set(dims_shape.keys()).intersection(da.coords.dims)
            da_co = {c: da[c].values for c in coords_labels}
            rec_da.loc[da_co] = rec_da.sel(da_co) + wda
            count_da.loc[da_co] = count_da.sel(da_co) + w

        final_reconstruction = (
            (rec_da / count_da)
            .to_dataset(name='sla')
            .isel(
                lat=slice(self.crop_val, -self.crop_val),
                lon=slice(self.crop_val, -self.crop_val),
            )
            .sel(lon=slice(-180, 180))
            .rename(lat="latitude", lon="longitude")
        )

        final_reconstruction = final_reconstruction.assign(
            longitude=lambda x: (x.longitude + 360) % 360
        ).sortby("longitude")

        final_reconstruction.latitude.attrs["units"] = "degrees_north"
        final_reconstruction.longitude.attrs["units"] = "degrees_east"

        path = self.save_dir.parents[0] / f"{self.kwargs['method_name']}.nc"
        final_reconstruction.to_netcdf(path)
        print(f'The reconstruction was saved at {path}')


class XrDataset(torch.utils.data.Dataset):
    def __init__(self, patcher, postpro_fns=(PredictItem._make,)):
        self.patcher = patcher
        self.postpro_fns = postpro_fns

    def __getitem__(self, idx):
        item = toolz.thread_first(
            self.patcher[idx].load(),
            *self.postpro_fns,
        )
        return item

    def __len__(self):
        return len(self.patcher)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


@hydra.main(config_path="config", config_name="inference", version_base="1.3")
def run(cfg):
    # data
    domain = {k: slice(*v) for k, v in cfg.params.domains.items()}
    obs = hydra.utils.call(cfg.observation.data).sel(domain)
    noise = cfg.observation.noise

    norm_stats = {}
    postpro_fns = {}
    patcher = xrpatcher.XRDAPatcher(
        da=obs,
        patches=cfg.params.patch_dims,
        strides=cfg.params.strides,
        check_full_scan=False,
    )
    m, s = (
        patcher.da.mean().item(),
        patcher.da.std().item(),
    )
    norm_stats = m, s
    postpro_fns = (
        lambda item: PredictItem._make((item.values.astype(np.float32),)),
        lambda item: item._replace(input=(item.input - m) / s),
    )

    datasets = XrDataset(patcher=patcher, postpro_fns=postpro_fns)
    dataloader = torch.utils.data.DataLoader(
        datasets,
        batch_size=cfg.params.get('batch_size', 1),
        num_workers=cfg.params.get('num_workers', 4),
    )

    resolution = (patcher.da.lat[1] - patcher.da.lat[0]).item()

    # model
    node = OmegaConf.select(OmegaConf.load(Path(cfg.method.config)), "model")
    checkpoint = torch.load(cfg.method.checkpoint, weights_only=True)["state_dict"]
    model = hydra.utils.call(node)
    model.load_state_dict(checkpoint)

    # other parameters
    ensemble = None
    if cfg.params.get('ensemble'):
        ensemble = {
            'size': int(cfg.params.ensemble),
            'noise': noise,
        }
        method_name = cfg.method.name + f'-size{cfg.params.ensemble}-std{noise}'
        print(f"Ensemble inference enabled: {ensemble}")
    else:
        method_name = cfg.method.name
        ensemble = None
        print("Ensemble inference disabled.")

    # Lightning Module and Trainer
    litmod = LitModel(
        patcher,
        model,
        norm_stats,
        save_dir=Path(cfg.method.checkpoint),
        method_name=method_name,
        crop_val=int(1 / resolution),
        out_coords={
            'time': pd.date_range(
                patcher.da.time[0].dt.date.item(),
                patcher.da.time[-1].dt.date.item(),
                freq="1D",
            ),
            'lat': np.arange(
                patcher.da.lat[0].item(),
                patcher.da.lat[-1].item() + resolution,
                resolution,
            ),
            'lon': np.arange(
                patcher.da.lon[0].item(),
                patcher.da.lon[-1].item() + resolution,
                resolution,
            ),
        },
        ensemble=ensemble,
    )

    trainer = pl.Trainer(
        inference_mode=False,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.predict(litmod, dataloader)
    print('> Inference finished.')


def build_weight(patch_dims, dim_weights=None):
    if not dim_weights:
        dim_weights = {'time': triang, 'lat': crop, 'lon': crop}

    return (
        dim_weights.get("time", np.ones)(patch_dims["time"])[:, None, None]
        * dim_weights.get("lat", np.ones)(patch_dims["lat"])[None, :, None]
        * dim_weights.get("lon", np.ones)(patch_dims["lon"])[None, None, :]
    )


def crop(n, crop_val):
    if crop_val:
        w = np.zeros(n)
        w[crop_val:-crop_val] = 1.0
        return w
    return np.ones(n)


def load_dataset(path, var, mask=None):
    if mask:
        ds = xr.open_dataset(path)
        return ds[var].where(~ds[mask]).astype(np.float64)
    return xr.open_dataset(path)[var].astype(np.float64)


def triang(n, min=0.05):
    return np.clip(1 - np.abs(np.linspace(-1, 1, n)), min, 1.0)


if __name__ == "__main__":
    run()
