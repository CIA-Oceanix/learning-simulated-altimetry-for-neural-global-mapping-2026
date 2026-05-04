import functools as ft
import numpy as np
import ocean4dvarnet


class NoisyLazyDataModule(ocean4dvarnet.data.LazyDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = {
            'train': np.random.default_rng(),
            'val': np.random.default_rng(333),
        }
        self.noise_level = .03  # in meters

    def post_fn(self, phase=None):
        m, s = self.norm_stats(phase)

        def add_noise(x):
            nl = self.noise_level

            if phase == 'train':
                scale = self._rng['train'].uniform(0., nl)
                noise = scale * self._rng['train'].normal(0., 1., x.shape)
            elif phase == 'val':
                noise = self._rng['val'].uniform(-nl, nl, x.shape)
            else:
                noise = 0.

            return x + noise.astype(np.float32)

        return ft.partial(ft.reduce, lambda i, f: f(i), [
            ocean4dvarnet.data.TrainingItem._make,
            lambda item: item._replace(tgt=(item.tgt - m) / s),
            lambda item: item._replace(input=(add_noise(item.input) - m) / s),
        ])
