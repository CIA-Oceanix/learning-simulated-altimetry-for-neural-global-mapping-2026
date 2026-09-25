import pathlib

import netCDF4
import numpy as np
import pytorch_lightning as pl
import scipy.signal
import torch
import xarray as xr
from ocean4dvarnet.data import TrainingItem
from torch.utils.data import DataLoader, Dataset

# Scalar fields available in The Well's HDF5 files, per dataset, as
# (group, variable, component indices). Tensor fields are stored with their
# components as trailing dimensions: (trajectory, time, x, y, *components).
FIELDS = {
    "active_matter": {
        "concentration": ("t0_fields", "concentration", ()),
        "velocity_x": ("t1_fields", "velocity", (0,)),
        "velocity_y": ("t1_fields", "velocity", (1,)),
        "D_xx": ("t2_fields", "D", (0, 0)),
        "D_xy": ("t2_fields", "D", (0, 1)),
        "D_yy": ("t2_fields", "D", (1, 1)),
        "E_xx": ("t2_fields", "E", (0, 0)),
        "E_xy": ("t2_fields", "E", (0, 1)),
        "E_yy": ("t2_fields", "E", (1, 1)),
    },
    "gray_scott_reaction_diffusion": {
        "A": ("t0_fields", "A", ()),
        "B": ("t0_fields", "B", ()),
    },
}


def open_file(path):
    try:
        return netCDF4.Dataset(path)
    except OSError as e:
        raise OSError(f"Cannot open {path} (incomplete download?)") from e


def list_trajectories(root, field):
    """List (file, trajectory index, number of time steps) in `root`."""
    group, name, _ = field
    trajectories = []
    for path in sorted(pathlib.Path(root).glob("*.hdf5")):
        with open_file(path) as nc:
            n_traj, n_time = nc[group][name].shape[:2]
        trajectories += [(path, traj, n_time) for traj in range(n_traj)]
    return trajectories


def fourier_resize(x, size):
    """Resize the last two (periodic) dimensions of `x` to `size` by Fourier
    interpolation: when upsampling, the original values are kept on the
    original grid points and no energy is added above the original Nyquist
    wavenumber."""
    for axis, n in zip((-2, -1), size):
        if x.shape[axis] != n:
            x = scipy.signal.resample(x, n, axis=axis)
    return x


class WellDataset(Dataset):
    """
    Extracts windows of `patch_dims["time"]` time steps (one out of
    `time_step`) from The Well's simulations and masks them with a real
    altimetry track mask, so that these physical fields go through the same
    masked-reconstruction pipeline as the SLA data.

    Each (trajectory, window start, field) triplet is a sample: the selected
    scalar fields are used as independent single-variable examples. Window
    starts are taken every `window_stride` time steps. Fields whose spatial
    size differs from the patch size are resized by Fourier interpolation
    when `resize` is "fourier" (The Well's domains are periodic).

    Noise follows `ocean4dvarnet.data.NoisyLazyDataModule`: it is added to the
    observations before normalisation, with a standardised gaussian noise
    scaled by U(0, noise) for training, a U(-noise, noise) noise for
    validation, and no noise for testing.

    Masks (and validation noise) are drawn at random for training, and from
    the sample index otherwise, so that validation scores are comparable
    across epochs.
    """

    def __init__(
        self,
        root,
        fields,
        mask_da,
        patch_dims,
        mean,
        std,
        noise=None,
        phase="train",
        time_step=1,
        window_stride=1,
        resize=None,
    ):
        self.fields = fields
        self.mask_da = mask_da
        self.n_frames = patch_dims["time"]
        self.size = (patch_dims["lat"], patch_dims["lon"])
        self.mean = mean
        self.std = std
        self.noise = noise
        self.phase = phase
        self.time_step = time_step
        self.resize = resize

        span = (self.n_frames - 1) * time_step + 1
        self.samples = [
            (path, traj, start, field)
            for path, traj, n_time in list_trajectories(root, fields[0])
            for start in range(0, n_time - span + 1, window_stride)
            for field in range(len(fields))
        ]

        if not self.samples:
            raise FileNotFoundError(f"No trajectory found in {root}")

        # netCDF4 handles cannot be shared across DataLoader worker processes:
        # they are opened lazily, in the process that reads them.
        self._files = {}
        self._rng_seed = None
        self._rng_train = None

    def _file(self, path):
        if path not in self._files:
            self._files[path] = open_file(path)
        return self._files[path]

    def _rng(self, idx):
        if self.phase != "train":
            return np.random.default_rng(idx)

        # One generator per DataLoader worker (torch seeds each worker
        # differently), so that workers do not draw the same masks.
        seed = torch.initial_seed()
        if self._rng_seed != seed:
            self._rng_seed = seed
            self._rng_train = np.random.default_rng(seed)
        return self._rng_train

    def _load_field(self, path, traj, start, field):
        group, name, components = self.fields[field]
        var = self._file(path)[group][name]
        var.set_auto_mask(False)
        stop = start + (self.n_frames - 1) * self.time_step + 1
        sl = (traj, slice(start, stop, self.time_step), slice(None), slice(None))
        x = np.asarray(var[sl + components], dtype=np.float32)

        if x.shape[-2:] != self.size:
            if self.resize != "fourier":
                raise ValueError(
                    f"Field of spatial size {x.shape[-2:]} does not match the "
                    + f"patch size {self.size}: set `resize: fourier`"
                )
            x = fourier_resize(x, self.size)
        return x

    def _sample_mask(self, rng):
        t_dim, lat_dim, lon_dim = self.mask_da.dims
        t0 = rng.integers(self.mask_da.sizes[t_dim])
        t_idx = (t0 + np.arange(self.n_frames)) % self.mask_da.sizes[t_dim]
        lat0 = rng.integers(self.mask_da.sizes[lat_dim] - self.size[0] + 1)
        lon0 = rng.integers(self.mask_da.sizes[lon_dim] - self.size[1] + 1)
        return self.mask_da.isel(
            {
                t_dim: t_idx,
                lat_dim: slice(lat0, lat0 + self.size[0]),
                lon_dim: slice(lon0, lon0 + self.size[1]),
            }
        ).values

    def _sample_noise(self, rng, shape):
        if not self.noise or self.phase == "test":
            return 0.0
        if self.phase == "train":
            return rng.uniform(0.0, self.noise) * rng.normal(0.0, 1.0, shape)
        return rng.uniform(-self.noise, self.noise, shape)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rng = self._rng(idx)
        tgt = self._load_field(*self.samples[idx])
        inp = np.where(self._sample_mask(rng), tgt, np.nan)
        inp = inp + self._sample_noise(rng, inp.shape)

        tgt = (tgt - self.mean) / self.std
        inp = (inp - self.mean) / self.std

        return TrainingItem(input=inp.astype(np.float32), tgt=tgt.astype(np.float32))


class WellDataModule(pl.LightningDataModule):
    """
    DataModule applying a real SLA altimetry track mask to one of The Well's
    datasets (see `FIELDS`), to train the same reconstruction models on
    physical data unrelated to oceanography.

    `path` must contain the `train`, `valid` and `test` subdirectories of the
    dataset, each holding HDF5 files.
    """

    def __init__(
        self,
        dataset,
        path,
        fields,
        mask_path,
        patch_dims,
        dl_kw,
        mask_var="inp",
        noise=None,
        time_step=1,
        window_strides=None,
        resize=None,
        norm_time_stride=20,
        **_ignored,
    ):
        # `**_ignored` absorbs the SLA-specific datamodule keys (input_da,
        # domains, xrds_kw, norm_stats) inherited from `unet`/`base.yaml`
        # through `defaults:`, which this datamodule does not use.
        super().__init__()
        self.path = pathlib.Path(path)
        self.fields = [FIELDS[dataset][f] for f in fields]
        self.mask_path = mask_path
        self.mask_var = mask_var
        self.patch_dims = patch_dims
        self.dl_kw = dl_kw
        self.noise = noise
        self.time_step = time_step
        self.window_strides = {"train": 1, "val": 1, "test": 1} | dict(window_strides or {})
        self.resize = resize
        self.norm_time_stride = norm_time_stride

        self._norm_stats = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def norm_stats(self):
        return self._norm_stats

    def train_mean_std(self):
        """Mean and std of the selected fields, on their native grid, over
        every training trajectory and one time step out of
        `norm_time_stride` (Fourier interpolation preserves both)."""
        count, total, total_sq = 0, 0.0, 0.0
        for path, traj, n_time in list_trajectories(self.path / "train", self.fields[0]):
            with open_file(path) as nc:
                for group, name, components in self.fields:
                    var = nc[group][name]
                    var.set_auto_mask(False)
                    sl = (traj, slice(0, n_time, self.norm_time_stride), slice(None), slice(None))
                    values = np.asarray(var[sl + components], dtype=np.float64)
                    count += values.size
                    total += values.sum()
                    total_sq += (values**2).sum()
        mean = total / count
        return mean, (total_sq / count - mean**2) ** 0.5

    def setup(self, stage=None):
        mask_da = xr.open_dataset(self.mask_path)[self.mask_var]

        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
        mean, std = self._norm_stats

        kw = dict(
            fields=self.fields,
            mask_da=mask_da,
            patch_dims=self.patch_dims,
            mean=mean,
            std=std,
            time_step=self.time_step,
            resize=self.resize,
        )
        self.train_ds = WellDataset(
            self.path / "train", noise=self.noise, phase="train",
            window_stride=self.window_strides["train"], **kw,
        )
        self.val_ds = WellDataset(
            self.path / "valid", noise=self.noise, phase="val",
            window_stride=self.window_strides["val"], **kw,
        )
        self.test_ds = WellDataset(
            self.path / "test", phase="test",
            window_stride=self.window_strides["test"], **kw,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
