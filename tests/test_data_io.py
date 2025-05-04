import pytest
import numpy as np
import pandas as pd
from windbem.data_io import BEMDataLoader

# ----------------------------------------
# Initialization and Data Structure Tests
# ----------------------------------------

# Title: Ensure BEMDataLoader initializes with correct default attributes and data types
def test_bem_data_loader_initialization(bem_data_loader):
    assert isinstance(bem_data_loader, BEMDataLoader)
    assert isinstance(bem_data_loader.blade_data, pd.DataFrame)
    assert isinstance(bem_data_loader.operational_data, pd.DataFrame)
    assert isinstance(bem_data_loader.polar_data, list)
    assert bem_data_loader.rho == 1.225
    assert bem_data_loader.blade_rad == 120
    assert bem_data_loader.no_blades == 3

# Title: Validate structure and required columns of loaded blade data
def test_load_blade_data(bem_data_loader):
    blade_data = bem_data_loader.blade_data
    assert isinstance(blade_data, pd.DataFrame)
    assert not blade_data.empty
    expected_columns = [
        'BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist',
        'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt'
    ]
    for col in expected_columns:
        assert col in blade_data.columns

# Title: Validate structure and required columns of loaded operational data
def test_load_operational_data(bem_data_loader):
    operational_data = bem_data_loader.operational_data
    assert isinstance(operational_data, pd.DataFrame)
    assert not operational_data.empty
    expected_columns = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
    for col in expected_columns:
        assert col in operational_data.columns

# Title: Validate that polar data contains required keys and correct types
def test_load_polar_data(bem_data_loader):
    polar_data = bem_data_loader.polar_data
    assert isinstance(polar_data, list)
    assert len(polar_data) > 0
    for airfoil in polar_data:
        assert 'af_index' in airfoil
        assert isinstance(airfoil['Alpha'], np.ndarray)
        assert isinstance(airfoil['Cl'], np.ndarray)
        assert isinstance(airfoil['Cd'], np.ndarray)
        assert isinstance(airfoil['Cm'], np.ndarray)

# Title: Check that airfoil shapes are correctly imported as coordinate arrays
def test_import_af_shapes(bem_data_loader):
    af_coords = bem_data_loader.import_af_shapes()
    assert isinstance(af_coords, list)
    assert len(af_coords) > 0
    for x, y in af_coords:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == len(y)

# Title: Ensure polar data is sorted by airfoil index
def test_polar_data_integrity(bem_data_loader):
    polar_data = bem_data_loader.polar_data
    af_indices = [airfoil['af_index'] for airfoil in polar_data]
    assert af_indices == sorted(af_indices)

# Title: Verify that airfoil plotting does not raise any exceptions
def test_plot_af_shapes(bem_data_loader):
    af_coords = bem_data_loader.import_af_shapes()
    try:
        bem_data_loader.plot_af_shapes(af_coords)
    except Exception as e:
        pytest.fail(f"Plotting airfoil shapes failed with error: {e}")

# ----------------------------------------
# Extended Functionality & Edge Case Tests
# ----------------------------------------

# Title: Confirm custom column names are used correctly in blade data loading
def test_load_blade_data_with_custom_columns(tmp_path):
    # Write blade data with 10 columns
    blade_file = tmp_path / "blade.dat"
    blade_file.write_text("Header\n" * 6 + " ".join([str(i) for i in range(10)]) + "\n" * 3)

    # Write dummy operational data with 5 columns to avoid crashing in init
    op_file = tmp_path / "op.dat"
    op_file.write_text("Header\n" + "1 2 3 4 5\n" * 3)

    custom_names = [f"col{i}" for i in range(10)]
    loader = BEMDataLoader(str(blade_file), str(op_file), str(tmp_path))
    df = loader._load_blade_data(str(blade_file), skiprow_num=6, aerodyn_names=custom_names)
    assert list(df.columns) == custom_names

# Title: Confirm custom column names are used correctly in operational data loading
def test_load_operational_data_with_custom_columns(tmp_path):
    file = tmp_path / "op.dat"
    file.write_text("Header\n" + "1 2 3 4 5\n" * 3)
    custom_names = [f"op_col{i}" for i in range(5)]
    loader = BEMDataLoader(str(file), str(file), str(tmp_path))
    df = loader._load_operational_data(str(file), skiprow_num=1, onshore_names=custom_names)
    assert list(df.columns) == custom_names

# Title: Ensure that malformed polar files are skipped without crashing
def test_load_polar_data_with_invalid_file(tmp_path):
    invalid_file = tmp_path / "badPolar_001.dat"
    invalid_file.write_text("Header\n" * 60 + "a b c d e\n")
    loader = BEMDataLoader(str(invalid_file), str(invalid_file), str(tmp_path))
    data = loader._load_polar_data(tmp_path)
    assert isinstance(data, list)
    assert len(data) == 0  # Should skip bad file

# Title: Verify correct coordinate extraction from a manually passed airfoil shape file
def test_import_af_shapes_with_explicit_paths(tmp_path):
    af_file = tmp_path / "AF_test.txt"
    af_file.write_text("\n" * 8 + "0.0 0.0\n1.0 1.0\n")
    loader = BEMDataLoader(str(af_file), str(af_file), str(tmp_path))
    af_coords = loader.import_af_shapes([af_file])
    assert isinstance(af_coords, list)
    assert len(af_coords) == 1
    x, y = af_coords[0]
    assert np.allclose(x, [0.0, 1.0])
    assert np.allclose(y, [0.0, 1.0])