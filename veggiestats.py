import xarray as xr
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box

plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#333333",
    "axes.labelcolor":      "#333333",
    "xtick.color":          "#333333",
    "ytick.color":          "#333333",
    "text.color":           "#333333",
    "font.family":          "sans-serif",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.titleweight":     "bold",
})

NDVI_FILE  = "2017.nc"
SHAPEFILE  = "nuts/NUTS_RG_20M_2024_4326.shp"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ndvi(filepath, engine="netcdf4"):
    raw_dataset = xr.open_dataset(filepath, engine=engine)
    raw_dataset = raw_dataset.isel(YDim=slice(None, None, -1))
    raw_dataset = raw_dataset.rename({"1 km monthly NDVI": "ndvi"})

    ndvi_with_spatial_ref = raw_dataset.copy()
    ndvi_with_spatial_ref = ndvi_with_spatial_ref.assign_coords({
        "x": (["XDim"], np.arange(raw_dataset.sizes["XDim"])),
        "y": (["YDim"], np.arange(raw_dataset.sizes["YDim"]))
    })

    lat_array  = raw_dataset["Latitude"].values
    lon_array  = raw_dataset["Longitude"].values
    ndvi_array = raw_dataset["ndvi"].values
    ndvi_array = np.where(ndvi_array > 10000, np.nan, ndvi_array / 10000.0)
    time_index = raw_dataset["time"].values

    print(f"NDVI Shape: {ndvi_array.shape}")
    print(f"Lat range:  {lat_array.min():.3f} – {lat_array.max():.3f}")
    print(f"Lon range:  {lon_array.min():.3f} – {lon_array.max():.3f}")
    print(f"NDVI range: {np.nanmin(ndvi_array):.3f} – {np.nanmax(ndvi_array):.3f}")
    print(f"Zeit:       {time_index}")

    return raw_dataset, lat_array, lon_array, ndvi_array, time_index


def load_and_reproject_shapefile(shapefile_path, target_crs="EPSG:4326"):
    admin_regions = gpd.read_file(shapefile_path)
    extent_box = box(0, 50, 15, 60)

    admin_regions = admin_regions[
        (admin_regions["LEVL_CODE"] == 2) &
        (admin_regions["CNTR_CODE"].isin(["DE", "DK", "NO", "SE"])) &
        (admin_regions.geometry.intersects(extent_box))
    ].copy()


    if admin_regions.crs is None:
        admin_regions = admin_regions.set_crs("EPSG:4326")

    if str(admin_regions.crs) != target_crs:
        admin_regions = admin_regions.to_crs(target_crs)
        print(f"Reprojiziert auf {target_crs}")

    return admin_regions


def build_pixel_geodataframe(lat_array, lon_array):
    num_y, num_x = lat_array.shape

    flat_lats = lat_array.flatten()
    flat_lons = lon_array.flatten()
    y_indices = np.repeat(np.arange(num_y), num_x)
    x_indices = np.tile(np.arange(num_x), num_y)

    pixel_frame = gpd.GeoDataFrame(
        {"y_idx": y_indices, "x_idx": x_indices},
        geometry=gpd.points_from_xy(flat_lons, flat_lats),
        crs="EPSG:4326"
    )

    return pixel_frame


def assign_pixels_to_regions(pixel_frame, admin_regions, region_name_column):
    pixels_with_region = gpd.sjoin(
        pixel_frame,
        admin_regions[[region_name_column, "geometry"]],
        how="left",
        predicate="within"
    )

    print(f"Regionen gefunden: {pixels_with_region[region_name_column].nunique()}")

    return pixels_with_region


def extract_regional_timeseries(ndvi_array, time_index, pixels_with_region, region_name_column):
    all_region_timeseries = []
    unique_regions = pixels_with_region[region_name_column].dropna().unique()

    for region_name in unique_regions:
        region_pixels = pixels_with_region[pixels_with_region[region_name_column] == region_name]
        y_positions   = region_pixels["y_idx"].values
        x_positions   = region_pixels["x_idx"].values

        for month_index, timestamp in enumerate(time_index):
            monthly_ndvi_slice = ndvi_array[month_index, :, :]
            pixel_ndvi_values  = monthly_ndvi_slice[y_positions, x_positions]
            valid_ndvi_values = pixel_ndvi_values[~np.isnan(pixel_ndvi_values)]
            valid_ndvi_values = valid_ndvi_values[(valid_ndvi_values >= 0) & (valid_ndvi_values <= 1)]
            if len(valid_ndvi_values) == 0:
                continue

            all_region_timeseries.append({
                "region":      region_name,
                "time":        pd.Timestamp(timestamp),
                "ndvi_mean":   float(np.mean(valid_ndvi_values)),
                "ndvi_median": float(np.median(valid_ndvi_values)),
                "ndvi_std":    float(np.std(valid_ndvi_values)),
                "ndvi_min":    float(np.min(valid_ndvi_values)),
                "ndvi_max":    float(np.max(valid_ndvi_values)),
                "pixel_count": int(len(valid_ndvi_values))
            })

    return pd.DataFrame(all_region_timeseries)


def plot_annual_ndvi_map(ndvi_array, lat_array, lon_array, output_path):
    with np.errstate(all="ignore"):
        annual_mean_ndvi = np.nanmean(ndvi_array, axis=0)
    annual_mean_ndvi[annual_mean_ndvi < -1] = np.nan

    _, ax = plt.subplots(figsize=(10, 9))

    scatter = ax.scatter(
        lon_array.flatten()[::4],
        lat_array.flatten()[::4],
        c=annual_mean_ndvi.flatten()[::4],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=0.85,
        s=2.5,
        linewidths=0
    )

    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("NDVI Jahresmittel 2017", fontsize=10)

    ax.set_xlabel("Längengrad")
    ax.set_xlim(0, 15)
    ax.set_ylabel("Breitengrad")
    ax.set_title("NDVI Jahresmittel 2017: Räumliche Verteilung")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Karte gespeichert: {output_path}")


def plot_regional_timeseries(regional_stats_frame, region_name_column, output_path, title="NDVI Regionale Zeitreihen"):
    unique_regions = regional_stats_frame["region"].unique()
    num_regions    = len(unique_regions)

    month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

    num_cols = 3
    num_rows = (num_regions + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(15, 4 * num_rows),
        sharey=True
    )
    axes_flat = axes.flatten() if num_regions > 1 else [axes]

    region_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]


    plot_index = 0
    for plot_index, region_name in enumerate(unique_regions):
        region_data = regional_stats_frame[
            regional_stats_frame["region"] == region_name
        ].sort_values("time")

        ax    = axes_flat[plot_index]
        color = region_colors[plot_index % len(region_colors)]

        ax.plot(
            range(len(region_data)),
            region_data["ndvi_mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
        )

        ax.set_title(region_name)
        ax.set_ylabel("NDVI")
        ax.set_xticks(range(len(region_data)))
        ax.set_xticklabels(month_labels[:len(region_data)], fontsize=8)
        ax.set_ylim(-0.1, 0.9)
        ax.set_xlim(0, 15)

    for hidden_index in range(plot_index + 1, len(axes_flat)):
        axes_flat[hidden_index].set_visible(False)

    fig.suptitle(title, fontsize=20, y=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zeitreihen gespeichert: {output_path}")


def plot_monthly_ndvi_maps(ndvi_array, lat_array, lon_array, time_index, output_path):
    month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    axes_flat = axes.flatten()

    valid_mask  = ndvi_array > -1
    global_vmin = np.nanpercentile(ndvi_array[valid_mask], 2)
    global_vmax = np.nanpercentile(ndvi_array[valid_mask], 98)

    last_scatter = None
    for month_index in range(12):
        ax = axes_flat[month_index]

        monthly_slice = ndvi_array[month_index].copy()
        monthly_slice[monthly_slice < -1] = np.nan

        last_scatter = ax.scatter(
            lon_array.flatten()[::9],
            lat_array.flatten()[::9],
            c=monthly_slice.flatten()[::9],
            cmap="RdYlGn",
            vmin=global_vmin,
            vmax=global_vmax,
            s=0.3,
            linewidths=0
        )

        ax.set_title(month_labels[month_index])
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 15)
    fig.colorbar(last_scatter, ax=axes, location="right", fraction=0.02, label="NDVI") # type: ignore
    fig.suptitle("NDVI 2017: Monatliche Übersicht", fontsize=20)

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Monatskarten gespeichert: {output_path}")


def run_full_pipeline(ndvi_filepath, shapefile_path, region_name_column="NAME_LATN"):
    print("Starte Pipeline...")
    print("=" * 60)

    print("\nLade NDVI Daten...")
    raw_dataset, lat_array, lon_array, ndvi_array, time_index = load_ndvi(ndvi_filepath)

    print("\nLade Shapefile...")
    admin_regions = load_and_reproject_shapefile(shapefile_path)

    print("\nBaue Pixel-Geodataframe und mappe auf Regionen...")
    pixel_frame        = build_pixel_geodataframe(lat_array, lon_array)
    pixels_with_region = assign_pixels_to_regions(pixel_frame, admin_regions, region_name_column)
    

    print("\nExtrahiere regionale Zeitreihen...")
    regional_stats_frame = extract_regional_timeseries(
        ndvi_array, time_index, pixels_with_region, region_name_column
    )
    
    csv_output_path = f"{OUTPUT_DIR}/ndvi_regional_stats.csv"
    regional_stats_frame.to_csv(csv_output_path, index=False)
    print(f"Stats CSV gespeichert: {csv_output_path}")
    print(regional_stats_frame.groupby("region")["ndvi_mean"].describe().round(3))

    print("\nErstelle Plots...")
    plot_annual_ndvi_map(
        ndvi_array, lat_array, lon_array,
        f"{OUTPUT_DIR}/ndvi_annual_map.png"
    )
    plot_monthly_ndvi_maps(
        ndvi_array, lat_array, lon_array, time_index,
        f"{OUTPUT_DIR}/ndvi_monthly_maps.png"
    )
    for country_code, country_name in [("DE", "Deutschland"), ("DK", "Dänemark"), ("NO", "Norwegen"), ("SE", "Schweden")]:
        country_regions = regional_stats_frame[
            regional_stats_frame["region"].isin(
                admin_regions[admin_regions["CNTR_CODE"] == country_code]["NAME_LATN"].values
                )
        ]
        if country_regions.empty:
            continue
        plot_regional_timeseries(
            country_regions,
            region_name_column,
            f"{OUTPUT_DIR}/ndvi_timeseries_{country_code}.png",
            title=f"NDVI Zeitreihe 2017 – {country_name}"
        )

    print("\n" + "=" * 60)
    print("All done!")

    return raw_dataset, admin_regions, regional_stats_frame


if __name__ == "__main__":
    raw_dataset, admin_regions, regional_stats = run_full_pipeline(
        ndvi_filepath=NDVI_FILE,
        shapefile_path=SHAPEFILE,
        region_name_column="NAME_LATN"
    )
