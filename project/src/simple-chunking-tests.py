import pytest
import xarray as xr
from pathlib import Path
from simple_benchmark import make_test_data, benchmark_read

def test_make_test_data(tmp_path):
    Path('./data').mkdir(exist_ok=True)
    chunks = (10, 18, 36)
    
    make_test_data(chunks)
    
    # Verify files exist and have correct structure
    ds_nc = xr.open_dataset(f'./data/temp_{chunks}.nc')
    ds_zarr = xr.open_zarr(f'./data/temp_{chunks}.zarr')
    
    assert ds_nc.temperature.shape == (1000, 180, 360)
    assert ds_zarr.temperature.shape == (1000, 180, 360)

def test_benchmark_read():
    chunks = (10, 18, 36)
    results = benchmark_read(chunks)
    
    assert 'netcdf' in results
    assert 'zarr' in results
    assert isinstance(results['netcdf'], float)
    assert isinstance(results['zarr'], float)
