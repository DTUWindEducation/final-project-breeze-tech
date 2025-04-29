import pytest
import numpy as np
import pandas as pd
from windbem.compute import BEMTurbineModel

# Test: Verify that the data is loaded correctly
def test_bem_data_loader(bem_model):
    assert bem_model.blade_data is not None, "Blade data not loaded!"
    assert bem_model.operational_data is not None, "Operational data not loaded!"
    assert bem_model.polar_data is not None, "Airfoil data not loaded!"
    
    assert isinstance(bem_model.blade_data, pd.DataFrame), "Blade data is not a DataFrame!"
    assert isinstance(bem_model.operational_data, pd.DataFrame), "Operational data is not a DataFrame!"
    assert isinstance(bem_model.polar_data, list), "Airfoil data is not a list!"

# Test: Verify the calculation of Cl and Cd
def test_get_cl_cd(bem_model):
    af_id = 1  # First airfoil
    alpha = 5.0  # Angle of attack in degrees
    
    cl, cd = bem_model.get_cl_cd(af_id, alpha)
    
    assert isinstance(cl, float), "Cl is not a float!"
    assert isinstance(cd, float), "Cd is not a float!"
    assert not np.isnan(cl), "Cl is NaN!"
    assert not np.isnan(cd), "Cd is NaN!"
    
    # Add boundary tests
    assert 0 <= cl <= 2, f"Cl out of range: {cl}"  # Cl typically between 0 and 2 for many airfoils
    assert 0 <= cd <= 1, f"Cd out of range: {cd}"  # Cd typically between 0 and 1

# Test: Verify that the induction coefficients (a, a_prime) are calculated correctly
def test_solve_bem_element(bem_model):
    r = 30.0  # Radial position (m)
    V0 = 10.0  # Wind speed (m/s)
    theta_p = 2.0  # Pitch angle (degrees)
    omega = 1.0  # Rotational speed (rad/s)
    
    result = bem_model.solve_bem_element(r, V0, theta_p, omega)
    
    # Verify that the results contain all the necessary calculated parameters
    assert all(k in result for k in ['a', 'a_prime', 'phi', 'alpha']), "Missing important results!"
    
    # Verify that the values are floats and not NaN
    assert isinstance(result['a'], float) and not np.isnan(result['a']), "Invalid value for 'a'!"
    assert isinstance(result['a_prime'], float) and not np.isnan(result['a_prime']), "Invalid value for 'a_prime'!"
    assert isinstance(result['phi'], float) and not np.isnan(result['phi']), "Invalid value for 'phi'!"
    assert isinstance(result['alpha'], float) and not np.isnan(result['alpha']), "Invalid value for 'alpha'!"
    
    # Add checks for the range of induction coefficients
    assert 0 <= result['a'] <= 1, f"a out of range: {result['a']}"
    assert 0 <= result['a_prime'] <= 1, f"a_prime out of range: {result['a_prime']}"

# Test: Verify rotor performance calculation
def test_compute_rotor_performance(bem_model):
    V0 = 10.0  # Wind speed (m/s)
    
    performance = bem_model.compute_rotor_performance(V0)
    
    assert 'thrust' in performance, "Thrust not calculated!"
    assert 'torque' in performance, "Torque not calculated!"
    assert 'power' in performance, "Power not calculated!"
    assert 'CT' in performance, "CT not calculated!"
    assert 'CP' in performance, "CP not calculated!"
    
    assert isinstance(performance['thrust'], float), "Thrust is not a float!"
    assert isinstance(performance['torque'], float), "Torque is not a float!"
    assert isinstance(performance['power'], float), "Power is not a float!"
    assert isinstance(performance['CT'], float), "CT is not a float!"
    assert isinstance(performance['CP'], float), "CP is not a float!"
    
    # Add checks for the range of CT and CP
    assert 0 <= performance['CT'] <= 1, f"CT out of range: {performance['CT']}"
    assert 0 <= performance['CP'] <= 0.6, f"CP out of range: {performance['CP']}"

# Test: Verify the power curve calculation
def test_compute_power_curve(bem_model):
    power_curve = bem_model.compute_power_curve()
    
    assert isinstance(power_curve, pd.DataFrame), "Power curve is not a DataFrame!"
    
    assert 'V0' in power_curve.columns, "Wind speed (V0) not found in power curve!"
    assert 'power' in power_curve.columns, "Power not found in power curve!"
    assert 'thrust' in power_curve.columns, "Thrust not found in power curve!"
    assert 'torque' in power_curve.columns, "Torque not found in power curve!"
    assert 'CT' in power_curve.columns, "CT not found in power curve!"
    assert 'CP' in power_curve.columns, "CP not found in power curve!"
    
    assert not power_curve.empty, "Power curve is empty!"

# Test: Verify handling of invalid input in get_cl_cd
def test_get_cl_cd_invalid_input(bem_model):
    # Test with an invalid airfoil ID (e.g., -1 or a non-existing ID)
    af_id = -1  # Invalid airfoil ID
    alpha = 5.0  # Angle of attack in degrees
    
    # Use pytest.raises to check if an exception is raised
    with pytest.raises(TypeError):
        # This should raise a TypeError since `polar_data` would be None or invalid
        bem_model.get_cl_cd(af_id, alpha)

# Test: Verify the integration of rotor results
def test_integrate_rotor_results(bem_model):
    V0 = 10.0  # Wind speed (m/s)
    performance = bem_model.compute_rotor_performance(V0)
    
    # Sum the contributions from each rotor element
    total_thrust_from_elements = sum(r['dT'] * r['dr'] for r in performance['element_results'])
    total_torque_from_elements = sum(r['dM'] * r['dr'] for r in performance['element_results'])
    
    # Verify that the calculated total is close to the integrated result
    assert np.isclose(total_thrust_from_elements, performance['thrust'], rtol=1e-2), "Inconsistent total thrust!"
    assert np.isclose(total_torque_from_elements, performance['torque'], rtol=1e-2), "Inconsistent total torque!"