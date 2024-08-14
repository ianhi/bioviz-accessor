from __future__ import annotations

import numpy as np
import xarray as xr
from dask.base import tokenize
from skimage import transform

__all__ = [
    "BioVizAccessor",
]


@xr.register_dataarray_accessor("bviz")
class BioVizAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._arr: xr.DataArray = xarray_obj
        self._center = None
        self._FOV_micron = self._arr.coords["X"].max()
        # protect against a supportable, but yet to be supported use case
        if (self._arr.coords["Y"].max() != self._FOV_micron) or (
            self._arr.shape[-1] != self._arr.shape[-2]
        ):
            raise ValueError("Non-square images are not yet supported")
        self._micron_to_pixel = (self._arr.shape[-1] / self._FOV_micron).values
        self._max_cache = 5
        self._token = None
        self._stitched_cache: dict[str, xr.DataArray] = {}

    @property
    def max_cache(self) -> int:
        return self._max_cache

    @max_cache.setter
    def max_cache(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("*max_cache* must be an int")
        self._max_cache = value
        while self._max_cache < len(self._stitched_cache):
            self._stitched_cache.pop(next(iter(self._stitched_cache)))

    def _validate_coords(self, name: str, value: int | None) -> int:
        if value is None:
            if self._arr.sizes[name] != 1:
                raise ValueError(
                    f"*{name}* must be specified when you"
                    " have more than one {name} coordinate."
                )
            value = 0
        return value

    def stitched(
        self, T: int | None = None, C: int | None = None, Z: int | None = None
    ) -> xr.DataArray:
        """
        Stitch together images from the array based on their coords.

        This function will use a cached version to avoid recomputation. However,
        the cache will be invalidated if you change the underlying array data. To avoid
        memory overflow there is also a cap on the size of cache. This is accessible and
        settable via the *max_cache* attribute which defualts to 5.

        Parameters
        ----------
        T, C, Z : int
            The coordinates of time, channel and Z slice to use. If the relevant
            dimension has size 1 then you may leave the argument as *None*. Otherwise
            these are required arguments.

        Returns
        -------
        xr.DataArray
            The stitched images as a single dataarray with updated coords for X and Y

        Note
        ----
        The final XY coords are slightly warped, however they are accurate enough for
        doing heuristic checks that your microscope experiment is in good working order.
        """
        Z = self._validate_coords("Z", Z)
        T = self._validate_coords("T", T)
        C = self._validate_coords("C", C)
        cache_key = f"{T}{C}{Z}"

        # we can't use something like LRU_cache here because we also need to check
        # the state of the underlying data which may have changed in the interim
        if self._token == tokenize(self._arr.data, ensure_determinisitic=True):
            # underlying array hasn't changed.
            # check if we've computed for these coords before
            if cache_key in self._stitched_cache:
                return self._stitched_cache[cache_key]
        else:
            self._stitched_cache = {}
            self._token = tokenize(self._arr.data, ensure_determinisitic=True)

        data = self._arr.sel(T=T, C=C, Z=Z)
        X_values = data.coords["Sx"] + data.coords["X"]
        Y_values = data.coords["Sy"] + data.coords["Y"]
        out_shape = (
            np.asarray(
                (Y_values.max() - Y_values.min(), X_values.max() - X_values.min())
            )
            * self._micron_to_pixel
        ).astype(int)

        # Do the actual image translating
        translated_imgs = []
        for i, arr in enumerate(data):
            tform = transform.AffineTransform(
                translation=(
                    np.array((data.coords["Sx"][i], data.coords["Sy"][i]))
                    - [
                        X_values.min(),
                        Y_values.min(),
                    ]  # offset by the farthest point in the data
                )
                * self._micron_to_pixel
            )
            out = transform.warp(
                arr, tform.inverse, output_shape=out_shape, mode="constant", cval=np.nan
            )
            translated_imgs.append(out)

        # np.linspace(Y_values.min(), Y_values.max())
        if len(self._stitched_cache) >= self._max_cache:
            # this removes the oldest dict item
            # this relies on python dicts being ordered
            # which they are since 3.6
            self._stitched_cache.pop(next(iter(self._stitched_cache)))

        self._stitched_cache[cache_key] = xr.DataArray(
            np.nanmean(translated_imgs, axis=0),
            dims=("Y", "X"),
            coords={
                # this is not quite right, it squishes or stretches espcially when there
                # is a long distance. But it's good enough for quick visualization
                # TODO: improve
                "Y": np.linspace(Y_values.min(), Y_values.max(), int(out_shape[0])),
                "X": np.linspace(X_values.min(), X_values.max(), int(out_shape[1])),
            },
        )

        return self._stitched_cache[cache_key]

    def hypersliced(self, **kwargs):  # type: ignore # noqa: D417 it is documented...
        """
        Display the area a 2D image with slider controlling remaining dimensions.

        Leverages ~mpl_interactions.hyperslicer~ to manage the display. If you
        are working in a notebook (e.g. jupyterlab) make sure to enable an interactive
        backend by using `%matplotlib ipympl`.

        You can control what figure this appears in by calling `plt.figure()`
        immediately prior to calling this function.

        Parameters
        ----------
        **kwargs : passed on to ~mpl_interactions.hyperslicer~

        Returns
        -------
        mpl_interactions.Controller
        """
        # lazy import so function is not requried
        from mpl_interactions import hyperslicer

        return hyperslicer(self._arr)
