"""Accessor for biology viz with xarray."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioviz_accessor")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Ian Hunt-Isaak"
__email__ = "ianhuntisaak@gmail.com"

from ._accessor import BioVizAccessor  # noqa: F401
