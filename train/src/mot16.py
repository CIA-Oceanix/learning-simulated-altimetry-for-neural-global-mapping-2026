import pathlib

import numpy as np
import pytorch_lightning as pl
import xarray as xr
from ocean4dvarnet.data import TrainingItem
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_sequence_frames(root):
    """List image frames, grouped by sequence sub-directory, in `root`."""
    return {
        sequence_dir: sorted(
            f for f in sequence_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for sequence_dir in sorted(pathlib.Path(root).iterdir())
        if sequence_dir.is_dir()
    }


def load_frame(path, size):
    """Load an image as a grayscale array of shape `(size[1], size[0])`, in [0, 1]."""
    img = Image.open(path).convert("L").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0


class ImageSequenceDataset(Dataset):
    """
    Extracts sliding windows of consecutive frames from image sequences
    (one subdirectory per sequence) and masks them with a real altimetry
    track mask, so that arbitrary image data can go through the same
    masked-reconstruction pipeline as the SLA data.
    """

    def __init__(self, root, mask_da, patch_dims, mean, std, noise=None):
        self.mask_da = mask_da
        self.n_frames = patch_dims["time"]
        self.size = (patch_dims["lon"], patch_dims["lat"])  # PIL wants (W, H)
        self.mean = mean
        self.std = std
        self.noise = noise

        self.windows = [
            frames[start : start + self.n_frames]
            for frames in list_sequence_frames(root).values()
            for start in range(len(frames) - self.n_frames + 1)
        ]

        if not self.windows:
            raise FileNotFoundError(f"No image sequence found in {root}")

    def _load_frame(self, path):
        return load_frame(path, self.size)

    def _sample_mask(self):
        t_dim, lat_dim, lon_dim = self.mask_da.dims
        t0 = np.random.randint(self.mask_da.sizes[t_dim])
        t_idx = (t0 + np.arange(self.n_frames)) % self.mask_da.sizes[t_dim]
        lat0 = np.random.randint(self.mask_da.sizes[lat_dim] - self.size[1] + 1)
        lon0 = np.random.randint(self.mask_da.sizes[lon_dim] - self.size[0] + 1)
        return self.mask_da.isel(
            {
                t_dim: t_idx,
                lat_dim: slice(lat0, lat0 + self.size[1]),
                lon_dim: slice(lon0, lon0 + self.size[0]),
            }
        ).values

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        tgt = np.stack([self._load_frame(f) for f in self.windows[idx]])
        inp = np.where(self._sample_mask(), tgt, np.nan)

        if self.noise:
            inp = inp + self.noise * np.random.normal(size=inp.shape)

        tgt = (tgt - self.mean) / self.std
        inp = (inp - self.mean) / self.std

        return TrainingItem(input=inp.astype(np.float32), tgt=tgt.astype(np.float32))


class MOT16DataModule(pl.LightningDataModule):
    """
    DataModule applying a real SLA altimetry track mask to sequences of
    arbitrary images (e.g. video frames), to train the same reconstruction
    models on data unrelated to geoscience.

    `path` must contain `train` and `val` subdirectories, each holding one
    subdirectory of ordered frames per sequence.
    """

    def __init__(
        self, path, mask_path, patch_dims, dl_kw, mask_var="inp", noise=None, **_ignored
    ):
        # `**_ignored` absorbs the SLA-specific datamodule keys (input_da,
        # domains, xrds_kw, norm_stats) inherited from `unet`/`base.yaml`
        # through `defaults:`, which this datamodule does not use.
        super().__init__()
        self.path = pathlib.Path(path)
        self.mask_path = mask_path
        self.mask_var = mask_var
        self.patch_dims = patch_dims
        self.dl_kw = dl_kw
        self.noise = noise

        self._norm_stats = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def norm_stats(self):
        return self._norm_stats

    def setup(self, stage=None):
        mask_da = xr.open_dataset(self.mask_path)[self.mask_var]
        size = (self.patch_dims["lon"], self.patch_dims["lat"])

        train_frames = [
            f
            for frames in list_sequence_frames(self.path / "train").values()
            for f in frames
        ]
        sample = train_frames[:: max(len(train_frames) // 500, 1)]
        pixels = np.stack([load_frame(f, size) for f in sample])
        mean, std = pixels.mean().item(), pixels.std().item()
        self._norm_stats = (mean, std)

        self.train_ds = ImageSequenceDataset(
            self.path / "train",
            mask_da,
            self.patch_dims,
            mean,
            std,
            noise=self.noise,
        )
        self.val_ds = ImageSequenceDataset(
            self.path / "val",
            mask_da,
            self.patch_dims,
            mean,
            std,
        )
        self.test_ds = ImageSequenceDataset(
            self.path / "val",
            mask_da,
            self.patch_dims,
            mean,
            std,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
