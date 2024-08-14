import numpy as np
import xarray as xr
from mda_simulator import ImageGenerator

__all__ = [
    "generate_fake_overlap_data",
]


def generate_fake_overlap_data(
    img_pixels: int = 512,
    FOV_micron: float = 100,
    overlap_fraction: float = 0.1,
    T: int = 2,
    Z: int = 3,
    C: int = 2,
) -> xr.DataArray:
    """
    Create fake partial tiling of microscope images.

    Parameters
    ----------
    img_pixels : int, default 512
        The edge length of the fake image
    FOV_micron : float, default 100
        The physical size of the image, nominally in microns.
    overlap_fraction : float
        A float in the range [0,1] controlling how much the images should overlap.
    T, Z, C : int
        The number of images along each dimension to generate.

    Returns
    -------
    xr.DataArray
    """
    image_shape = (img_pixels, img_pixels)
    gen = ImageGenerator(N=5000, img_shape=image_shape)

    XY_positions = np.asarray(
        [
            (0, 0),
            (FOV_micron * (1 - overlap_fraction), 0),
            (100 * (1 - overlap_fraction), 0),
            (FOV_micron * (1 - overlap_fraction), FOV_micron * 2),
            (FOV_micron * (1 + overlap_fraction), FOV_micron * (1 - overlap_fraction)),
            # (overlap_micron, -overlap_micron),
        ]
    )

    S = len(XY_positions)
    data = xr.DataArray(
        np.zeros((S, T, C, Z, *image_shape), dtype=np.uint16),
        dims=("S", "T", "C", "Z", "Y", "X"),
        coords={
            "Sx": ("S", XY_positions[:, 0]),
            "Sy": ("S", XY_positions[:, 1]),
            "X": np.linspace(0, FOV_micron, img_pixels),
            "Y": np.linspace(0, FOV_micron, img_pixels),
        },
    )
    for t in range(T):
        for s in range(S):
            for c in range(C):
                for z in range(Z):
                    data[s, t, c, z] = gen.snap_img(
                        XY_positions[s], c=c, z=z, exposure=5
                    )
        gen.increment_time(10)
    return data
