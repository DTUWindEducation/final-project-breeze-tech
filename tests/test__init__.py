import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from windbem.compute import BEMTurbineModel
from windbem import solve_bem_element, compute_rotor_performance, compute_power_curve, plot_results, print_results

# Test BEMTurbineModel instance (uses the bem_model fixture)
def test_bem_model(bem_model):
    assert isinstance(bem_model, BEMTurbineModel)

# Test solve_bem_element
def test_solve_bem_element(bem_model):
    # Choose a test radial position, wind speed, pitch angle, and rotational speed
    r = 30.0  # in meters
    v0 = 10.0  # wind speed in m/s
    theta_p = 5.0  # pitch angle in degrees
    omega = 1.2  # rotational speed in rad/s
    
    result = solve_bem_element(bem_model, r, v0, theta_p, omega)

    # Check that the returned result is a dictionary and contains the expected keys
    assert isinstance(result, dict)
    assert 'r' in result
    assert 'a' in result
    assert 'a_prime' in result
    assert 'phi' in result
    assert 'alpha' in result
    assert 'coef_lift' in result
    assert 'coef_drag' in result
    assert 'coef_normal' in result
    assert 'coef_tangent' in result
    assert 'd_thrust' in result
    assert 'd_torque' in result

# Test compute_rotor_performance
def test_compute_rotor_performance(bem_model):
    v0 = 10.0  # wind speed in m/s
    theta_p = 5.0  # pitch angle in degrees
    omega = 1.2  # rotational speed in rad/s

    performance = compute_rotor_performance(bem_model, v0, theta_p, omega)
    
    # Check that the returned result is a dictionary and contains the expected keys
    assert isinstance(performance, dict)
    assert 'v0' in performance
    assert 'theta_p' in performance
    assert 'omega' in performance
    assert 'thrust' in performance
    assert 'torque' in performance
    assert 'power' in performance
    assert 'thrust_coef' in performance
    assert 'power_coef' in performance
    assert 'element_results' in performance

# Test compute_power_curve
def test_compute_power_curve(bem_model):
    power_curve_df = compute_power_curve(bem_model)
    
    # Check that the result is a pandas DataFrame
    assert isinstance(power_curve_df, pd.DataFrame)
    
    # Check if the dataframe has expected columns
    expected_columns = ['v0', 'power', 'thrust', 'torque', 'thrust_coef', 'power_coef', 'pitch', 'omega']
    for col in expected_columns:
        assert col in power_curve_df.columns

def test_plot_and_print_results(bem_model, capsys):
    # Test plotting function
    try:
        plot_results(bem_model)
    except Exception as e:
        pytest.fail(f"plot_results raised an error: {e}")
    plt.close('all')  # Ensure plot is closed after testing

    # Test print_results
    v0 = 10.0  # wind speed in m/s
    performance = compute_rotor_performance(bem_model, v0)
    
    # Capture printed output
    print_results(performance)
    captured = capsys.readouterr()

    # Check if the printed results include expected information
    assert f"At {performance['v0']} m/s wind speed:" in captured.out
    assert f"Power: {performance['power']/1e3:.2f} kW" in captured.out
    assert f"Thrust: {performance['thrust']/1e3:.2f} kN" in captured.out
    assert f"Power coefficient: {performance['power_coef']:.3f}" in captured.out

