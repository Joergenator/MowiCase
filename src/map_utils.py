"""Lightweight cartopy helpers for plotting Norwegian-coast site maps.

Extracted from `notebooks/04_extra_charts.py` so multiple notebooks can share
the same basemap styling without re-running 04's heavy module-level code.

This module imports cartopy lazily — it's only required by callers that
actually invoke `add_basemap`. The rest of the codebase doesn't depend on it.
"""
from __future__ import annotations


def add_basemap(ax, extent: tuple[float, float, float, float]) -> None:
    """Draw Natural Earth ocean / land / coastline / borders on a cartopy axis.

    Parameters
    ----------
    ax
        A matplotlib axis created with a cartopy projection (e.g. via
        `subplot_kw={"projection": MAP_CRS}`).
    extent
        (lon_min, lon_max, lat_min, lat_max), passed in PlateCarree degrees.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#dfeaf2", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#f3efe6", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.5, edgecolor="#666666", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.4, edgecolor="#888888", linestyle=":",
                   zorder=2)


def get_crs():
    """Return (DATA_CRS, MAP_CRS) — PlateCarree and Mercator centered on Norway.

    Imported lazily so this module can be loaded without cartopy installed.
    Callers that only need lat/lon math don't pay the import cost.
    """
    import cartopy.crs as ccrs
    return ccrs.PlateCarree(), ccrs.Mercator(central_longitude=15)
