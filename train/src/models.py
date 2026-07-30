import functools as ft
import numpy as np
import ocean4dvarnet
import xarray as xr


class MaskingDataModule(ocean4dvarnet.data.NoisyLazyDataModule):
    def setup(self, stage=None):
        self.train_ds = MaskingLazyXrDataset(
            das={
                'tgt': self.input_da['tgt'].sel(self.domains['train']),
                'input': self.input_da['input'],
            },
            **self.xrds_kw["train"], postpro_fn=self.post_fn('train'),
        )
        self.val_ds = MaskingLazyXrDataset(
            das={
                'tgt': self.input_da['tgt'].sel(self.domains['val']),
                'input': self.input_da['input'],
            },
            **self.xrds_kw["val"], postpro_fn=self.post_fn('val'),
        )
        self.test_ds = MaskingLazyXrDataset(
            das={
                'tgt': self.input_da['tgt'].sel(self.domains['test']),
                'input': self.input_da['input'],
            },
            **self.xrds_kw["test"], postpro_fn=self.post_fn('test'),
        )


class MaskingLazyXrDataset(ocean4dvarnet.data.LazyXrDataset):
    def __init__(
        self,
        das,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        **kwargs,
    ):
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self._mask_size = das["input"].shape[0]
        self.da = {
            "tgt": das["tgt"].sel(domain_limits),
            "input": das["input"]
            .sel({k: v for (k, v) in domain_limits.items() if k != "time"})
            .assign_coords(time=range(self._mask_size)),
        }
        self._check_dims_and_coords()
        self.patch_dims = patch_dims
        self.strides = strides or {}
        ref = next(iter(self.da))
        da_dims = dict(zip(self.da[ref].dims, self.da[ref].shape))
        self.ds_size = {
            dim: max(
                (da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0
            )
            for dim in patch_dims
        }

        # If an edge-behaviour is specified for a dimension, increment
        # by one self.ds_size along this dimension
        self.edges = kwargs.get("edges", {})
        for dim, behaviour in self.edges.items():
            if behaviour not in ("periodic", "fill_nan"):
                raise ValueError(
                    f"edges[{dim}] must be either 'periodic' or 'fill_nan', "
                    + f"got {behaviour}"
                )

            if (da_dims[dim] - patch_dims[dim]) % strides[dim] != 0:
                self.ds_size[dim] += 1

    def __getitem__(self, item):
        sl = {}
        _zip = zip(
            self.ds_size.keys(),
            np.unravel_index(item, tuple(self.ds_size.values())),
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )

        sl_time = sl.pop("time")

        start = sl_time.start % self._mask_size
        stop = sl_time.stop % self._mask_size
        if start > stop:
            start -= stop
            stop = None
        sl_time_mask = slice(start, stop)

        ref = next(iter(self.da))
        sliced_domain = self.da[ref].isel({"time": sl_time} | sl)

        if self.return_coords:
            item = sliced_domain
            return item.coords.to_dataset()[list(self.patch_dims)]

        das = {"tgt": self.da["tgt"].isel(time=sl_time)}
        das["input"] = das["tgt"].where(self.da["input"].isel(time=sl_time_mask).data)

        # Handling edge behaviour
        for dim, behaviour in self.edges.items():
            if len(sliced_domain[dim]) >= self.patch_dims[dim]:
                continue
            offset = self.patch_dims[dim] - len(sliced_domain[dim])

            if behaviour == "periodic":
                sl[dim] = slice(
                    sl[dim].start - offset,
                    sl[dim].stop - offset,
                )

                for var in das:
                    das[var] = das[var].roll({dim: -offset})
            elif behaviour == "fill_nan":
                for var in das:
                    das[var] = das[var].pad(
                        pad_width={dim: (0, offset)},
                        mode="constant",
                        constant_values=np.nan,
                    )

        item = (
            xr.Dataset(
                data_vars=das,
                coords=next(iter(das.values())).coords,
            )
            .isel(sl)
            .to_dataarray()
            .sortby("variable")
            .data.astype(np.float32)
        )

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

    def _check_dims_and_coords(self):
        ref = next(iter(self.da))
        ref_val = self.da[ref]

        for k, v in self.da.items():
            if ref == k:
                continue

            if ref_val.dims != v.dims:
                raise ValueError(
                    "All provided xr.DataArray must share the same dimensions "
                    + f"({ref} != {k})"
                )

            a = ref_val.drop_vars("time").coords
            b = v.drop_vars("time").coords
            if not a.equals(b):
                raise ValueError(
                    "All provided xr.DataArray must share the same coordinates "
                    + f"({ref} != {k})"
                )


def load_sea_level_anomaly_with_mask(
    tgt_path, inp_path, tgt_var="sla", inp_var="sla",
):
    def rename_coords(ds):
        rename = {}

        if 'latitude' in ds.coords and 'lat' not in ds.coords:
            rename['latitude'] = 'lat'
        if 'longitude' in ds.coords and 'lon' not in ds.coords:
            rename['longitude'] = 'lon'

        return ds.rename(rename)

    tgt = rename_coords(
        xr.open_dataset(tgt_path)
        .rename(latitude="lat", longitude="lon")
    )[tgt_var]

    inp = rename_coords(xr.open_dataset(inp_path))[inp_var]

    return {"input": inp, "tgt": tgt}
