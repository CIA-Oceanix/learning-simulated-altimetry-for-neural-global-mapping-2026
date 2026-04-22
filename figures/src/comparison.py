from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

plt.rcParams['font.size'] = '16'

def plot_error(
    fig, ax, study, reference, title,
    variable='rmse',  # variable='variance_mapping_err',
    all_scale=False, lon_boundary=None, lat_boundary=None, show_cbar=True,
    show_axis_legend=None,
):
    _scale = 'all_scale' if all_scale else 'filtered'
    if not lon_boundary:
        lon_boundary = (-180, 180)
    if not lat_boundary:
        lat_boundary = (-90, 90)

    ds_all_scale = xr.open_dataset(reference, group='all_scale')
    ds_ref_binning = xr.open_dataset(reference, group=_scale)
    ds_study_binning = xr.open_dataset(study, group=_scale)

    if all_scale:
        vmin = -2
        vmax = 2
    else:
        vmin = -4
        vmax = 4
    p0 = ax.pcolormesh(
        ds_all_scale.lon,
        ds_all_scale.lat,
        (ds_study_binning[variable] - ds_ref_binning[variable]) * 100,
        vmin=vmin,
        vmax=vmax,
        cmap="bwr",
    )
    # ax.set_title(f'{title} [{_scale}]')
    ax.coastlines(resolution="10m", lw=0.5)
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k")
    ax.set_extent(
        [lon_boundary[0], lon_boundary[-1], lat_boundary[0], lat_boundary[-1]],
        ccrs.PlateCarree(),
    )

    p0.axes.gridlines(color="black", alpha=0.0, linestyle="--")

    # draw parallels/meridiens and write labels
    gl = p0.axes.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.1,
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    # adjust labels to taste
    show_axis_legend = show_axis_legend or [True, True, True, True]
    gl.top_labels = show_axis_legend[0]
    gl.right_labels = show_axis_legend[1]
    gl.bottom_labels = show_axis_legend[2]
    gl.left_labels = True  # show_axis_legend[3]
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 12, "color": "black"}
    gl.ylabel_style = {"size": 12, "color": "black"}

    if show_cbar:
        cbar = fig.colorbar(
            p0, orientation="horizontal", extend="both", anchor=(0.5, 1.5), shrink=0.7
        )
        cbar.set_label("Gain (-) / Loss (+) in RMSE (cm)")

    return fig, ax

def plot_resolution(
    fig, ax, study, reference, title, variable='effective_resolution',
    in_percentage=True, lon_boundary=None, lat_boundary=None, show_cbar=True,
    show_axis_legend=None,
):
    if not lon_boundary:
        lon_boundary = (-180, 180)
    if not lat_boundary:
        lat_boundary = (-90, 90)

    ds_ref = xr.open_dataset(reference)
    ds_study = xr.open_dataset(study)

    difference = (ds_study[variable] - ds_ref[variable])
    # Comparison in % or in km
    if in_percentage:
        vmin, vmax = -10, 10
        difference = 100 * difference / ds_ref[variable]
    else:
        vmin, vmax = -40, 40
    p0 = ax.pcolormesh(
        ds_ref.lon,
        ds_ref.lat,
        difference,
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
    )
    # ax.set_title(title)
    ax.coastlines(resolution="10m", lw=0.5, zorder=13)
    ax.add_feature(cfeature.LAND, zorder=12, color="w")
    ax.set_extent(
        [lon_boundary[0], lon_boundary[-1], lat_boundary[0], lat_boundary[-1]],
        ccrs.PlateCarree(),
    )

    p0.axes.gridlines(color="black", alpha=0.0, linestyle="--")

    # draw parallels/meridiens and write labels
    gl = p0.axes.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.1,
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    # adjust labels to taste
    show_axis_legend = show_axis_legend or [True, True, True, True]
    gl.top_labels = show_axis_legend[0]
    gl.right_labels = True  # show_axis_legend[1]
    gl.bottom_labels = show_axis_legend[2]
    gl.left_labels = show_axis_legend[3]
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 12, "color": "black"}
    gl.ylabel_style = {"size": 12, "color": "black"}

    if show_cbar:
        cbar = fig.colorbar(
            p0, orientation="horizontal", extend="both", anchor=(0.5, 1.5), shrink=0.7
        )
        unit = "%" if in_percentage else "km"
        cbar.set_label(f"Gain (-) / Loss (+) in effective resolution ({unit})")

    return fig, ax

def plot_rmse_resolution(
    title, model_file, model_ref='miost', save=True, show_cbar=True,
    show_axis_legend=None, suptitley=.9, figure_name_prefix='',
    resources_path='./resources', target_path='./images', dpi=600,
    image_format='png',
):
    Path(target_path).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        nrows=1, ncols=2, subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(12, 5),
    )

    plot_error(
        fig, ax[0],
        f"{resources_path}/stat_sla_{model_file}.nc",
        f"{resources_path}/stat_sla_{model_ref}.nc",
        "RMSE",
        all_scale=True, show_cbar=show_cbar, show_axis_legend=show_axis_legend,
    )

    plot_resolution(
        fig, ax[1],
        f"{resources_path}/psd_sla_{model_file}.nc",
        f"{resources_path}/psd_sla_{model_ref}.nc",
        "Effective Resolution",
        in_percentage=False, show_cbar=show_cbar,
        show_axis_legend=show_axis_legend,
    )

    fig.suptitle(title, y=suptitley)
    fig.tight_layout()
    if save:
        fig.savefig(
            f'{target_path}/{figure_name_prefix}_{model_file}_{model_ref}.{image_format}',
            dpi=dpi,
        )
