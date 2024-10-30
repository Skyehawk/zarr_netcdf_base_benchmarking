#!/usr/bin/env python3
"""Simple script to demonstrate dask performance reports."""

import numpy as np
import xarray as xr
import pandas as pd
from dask.distributed import Client, performance_report
import multiprocessing

def main():
    # Create sample dataset (365 days x 100 lat x 100 lon)
    ds = xr.Dataset(
        data_vars={
            'temperature': (
                ['time', 'lat', 'lon'],
                np.random.randn(365, 100, 100)
            )
        },
        coords={
            'time': pd.date_range('2020-01-01', periods=365),
            'lat': np.linspace(-89.5, 89.5, 100),
            'lon': np.linspace(-179.5, 179.5, 100)
        }
    )

    # Set up dask client
    client = Client(n_workers=2, threads_per_worker=1, scheduler_port=0)
    print(f"\nDask dashboard available at: {client.dashboard_link}\n")

    try:
        # Chunk the data and perform calculation
        chunked_ds = ds.chunk({'time': 50, 'lat': 25, 'lon': 25})
        
        print("Computing mean temperature with performance report...")
        with performance_report(filename="dask-report.html"):
            result = chunked_ds.temperature.mean(['time', 'lat', 'lon']).compute()
        
        print(f"Mean temperature: {float(result.values):.4f}")
        print("\nPerformance report saved to: dask-report.html")

    finally:
        client.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
