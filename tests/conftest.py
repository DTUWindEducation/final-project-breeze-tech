import pytest
from windbem.compute import BEMTurbineModel
from windbem.data_io import BEMDataLoader

@pytest.fixture
def bem_model():
    # Paths to the required data files
    blade_file = 'inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat'
    operational_file = 'inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt'
    polar_dir = 'inputs/IEA-15-240-RWT/Airfoils/'
    
    # Initialize the BEMTurbineModel object
    return BEMTurbineModel(blade_file, operational_file, polar_dir)


@pytest.fixture
def bem_data_loader():
    # Paths to the required data files
    blade_file = 'inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat'
    operational_file = 'inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt'
    polar_dir = 'inputs/IEA-15-240-RWT/Airfoils/'
    
    # Initialize the BEMDataLoader object
    return BEMDataLoader(blade_file, operational_file, polar_dir)
