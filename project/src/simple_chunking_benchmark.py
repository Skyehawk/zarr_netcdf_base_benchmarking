import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, performance_report


def make_test_data(chunks: Dict[str, int]) -> None:
    """Generate sample dataset and save as both NetCDF and Zarr."""
    # Make synthetic data
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["time", "lat", "lon"],
                np.random.randn(365, 180, 360).astype("float32"),
                {"units": "celsius"},
            )
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=365, freq="D"),
            "lat": np.linspace(-89.5, 89.5, 180),  # 180 for single degree grid
            "lon": np.linspace(-179.5, 179.5, 360),  # 360 for single degree grid
        },
    )

    # Save both formats
    chunk_id = f"{chunks['time']}x{chunks['lat']}x{chunks['lon']}"
    Path("./data").mkdir(exist_ok=True)
    ds.chunk(chunks).to_netcdf(f"./data/temp_{chunk_id}.nc")
    ds.chunk(chunks).to_zarr(f"./data/temp_{chunk_id}.zarr", mode="w")


def benchmark_read(chunks: Dict[str, int]) -> Dict[str, float]:
    """Run benchmark on both formats with given chunking."""
    client = Client(n_workers=4, threads_per_worker=2, memory_limit="4GB")
    results = {}

    chunk_id = f"{chunks['time']}x{chunks['lat']}x{chunks['lon']}"

    # Test both formats
    for fmt in ["netcdf", "zarr"]:
        if fmt == "netcdf":
            ds = xr.open_dataset(f"./data/temp_{chunk_id}.nc", chunks=chunks)
        else:
            ds = xr.open_zarr(f"./data/temp_{chunk_id}.zarr")

        # Time the operation
        start = time.perf_counter()
        with performance_report(filename=f"dask-report-{fmt}-{chunk_id}.html"):
            result = (
                ds.temperature.isel(lat=slice(45, 90), lon=slice(90, 180))
                .sum("time")
                .compute()
            )

        results[fmt] = time.perf_counter() - start

    client.close()
    return results


if __name__ == "__main__":
    # Define chunking strategies
    strategies = [
        {"time": 365, "lat": 1, "lon": 1},  # time-optimized
        {"time": 1, "lat": 180, "lon": 360},  # spatial-optimized
        {"time": 100, "lat": 45, "lon": 90},  # balanced
        {"time": 10, "lat": 18, "lon": 36},  # tiny-chunks
    ]

    # Run benchmarks
    for chunks in strategies:
        print(f"\nTesting chunks {chunks}")
        make_test_data(chunks)
        times = benchmark_read(chunks)
        print(f"NetCDF time: {times['netcdf']:.2f}s")
        print(f"Zarr time: {times['zarr']:.2f}s")
