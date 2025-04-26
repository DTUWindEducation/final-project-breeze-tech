import pytest
import numpy as np
import pandas as pd
from windbem.data_io import BEMDataLoader

# Test: BEMDataLoader Initialization
def test_bem_data_loader_initialization(bem_data_loader):
    assert isinstance(bem_data_loader, BEMDataLoader)
    assert isinstance(bem_data_loader.blade_data, pd.DataFrame), "Blade data not loaded correctly!"
    assert isinstance(bem_data_loader.operational_data, pd.DataFrame), "Operational data not loaded correctly!"
    assert isinstance(bem_data_loader.polar_data, list), "Polar data not loaded correctly!"
    assert bem_data_loader.rho == 1.225, "Default air density value is incorrect!"
    assert bem_data_loader.blade_rad == 120, "Default blade radius is incorrect!"
    assert bem_data_loader.no_blades == 3, "Default number of blades is incorrect!"

# Test: Load blade data structure and check key columns
def test_load_blade_data(bem_data_loader):
    blade_data = bem_data_loader.blade_data
    assert isinstance(blade_data, pd.DataFrame)
    assert not blade_data.empty, "Blade data is empty!"
    expected_columns = [
        'BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist',
        'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt'
    ]
    for col in expected_columns:
        assert col in blade_data.columns, f"Missing column {col} in blade data!"

# Test: Load operational data structure and check key columns
def test_load_operational_data(bem_data_loader):
    operational_data = bem_data_loader.operational_data
    assert isinstance(operational_data, pd.DataFrame)
    assert not operational_data.empty, "Operational data is empty!"
    expected_columns = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
    for col in expected_columns:
        assert col in operational_data.columns, f"Missing column {col} in operational data!"

# Test: Load polar data structure and basic checks
def test_load_polar_data(bem_data_loader):
    polar_data = bem_data_loader.polar_data
    assert isinstance(polar_data, list)
    assert len(polar_data) > 0, "Polar data is empty!"
    for airfoil in polar_data:
        assert 'af_index' in airfoil
        assert isinstance(airfoil['Alpha'], np.ndarray)
        assert isinstance(airfoil['Cl'], np.ndarray)
        assert isinstance(airfoil['Cd'], np.ndarray)
        assert isinstance(airfoil['Cm'], np.ndarray)

# Test: Import and validate airfoil coordinates structure
def test_import_af_shapes(bem_data_loader):
    af_coords = bem_data_loader.import_af_shapes()
    assert isinstance(af_coords, list), "Airfoil coordinates should be a list!"
    assert len(af_coords) > 0, "No airfoil coordinates loaded!"
    for x, y in af_coords:
        assert isinstance(x, np.ndarray), f"X coordinates should be of type np.ndarray!"
        assert isinstance(y, np.ndarray), f"Y coordinates should be of type np.ndarray!"
        assert len(x) == len(y), "Mismatch between x and y coordinate lengths!"

# Test: Polar data integrity (ensure data is sorted by airfoil index)
def test_polar_data_integrity(bem_data_loader):
    polar_data = bem_data_loader.polar_data
    assert len(polar_data) > 0, "Polar data is empty!"
    # Ensure data is sorted by af_index
    af_indices = [airfoil['af_index'] for airfoil in polar_data]
    assert af_indices == sorted(af_indices), "Polar data is not sorted by airfoil index!"

# Test: Plotting airfoil shapes (check that plotting does not raise errors)
def test_plot_af_shapes(bem_data_loader):
    af_coords = bem_data_loader.import_af_shapes()
    try:
        bem_data_loader.plot_af_shapes(af_coords)
    except Exception as e:
        pytest.fail(f"Plotting airfoil shapes failed with error: {e}")
