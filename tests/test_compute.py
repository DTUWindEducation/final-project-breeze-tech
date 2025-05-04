import pytest
import numpy as np
import pandas as pd

# Test: Verify that the BEM model loads blade, operational, and polar data correctly
def test_bem_data_loader(bem_model):
    assert bem_model.blade_data is not None, "Blade data not loaded!"
    assert bem_model.operational_data is not None, "Operational data not loaded!"
    assert bem_model.polar_data is not None, "Airfoil data not loaded!"
    
    assert isinstance(bem_model.blade_data, pd.DataFrame), "Blade data is not a DataFrame!"
    assert isinstance(bem_model.operational_data, pd.DataFrame), "Operational data is not a DataFrame!"
    assert isinstance(bem_model.polar_data, list), "Airfoil data is not a list!"

# Test: Verify that lift and drag coefficients (Cl and Cd) are correctly computed for valid input
def test_get_cl_cd(bem_model):
    af_id = 1
    alpha = 5.0
    cl, cd = bem_model.get_cl_cd(af_id, alpha)
    
    assert isinstance(cl, float), "Cl is not a float!"
    assert isinstance(cd, float), "Cd is not a float!"
    assert not np.isnan(cl), "Cl is NaN!"
    assert not np.isnan(cd), "Cd is NaN!"
    assert 0 <= cl <= 2, f"Cl out of range: {cl}"
    assert 0 <= cd <= 1, f"Cd out of range: {cd}"

# Test: Verify that invalid airfoil ID raises a TypeError in get_cl_cd
def test_get_cl_cd_invalid_input(bem_model):
    af_id = -1
    alpha = 5.0
    with pytest.raises(TypeError):
        bem_model.get_cl_cd(af_id, alpha)

# Test: Verify that the operational strategy returns valid pitch and omega for various wind speeds
def test_get_operational_strategy(bem_model):
    v0 = 8.0
    theta_p, omega = bem_model.get_operational_strategy(v0)
    assert isinstance(theta_p, float)
    assert isinstance(omega, float)
    assert omega > 0

    v0_low = 3.0
    v0_high = 25.0
    theta_p_low, omega_low = bem_model.get_operational_strategy(v0_low)
    theta_p_high, omega_high = bem_model.get_operational_strategy(v0_high)
    assert theta_p_low > 0
    assert omega_low > 0
    assert theta_p_high > 0
    assert omega_high > 0

# Test: Verify that blade element span positions are returned as a positive numpy array
def test_get_element_spans(bem_model):
    spans = bem_model.get_element_spans()
    assert isinstance(spans, np.ndarray)
    assert len(spans) > 0
    assert np.all(spans > 0)


# Test: Validate correct computation of flow angle from wind, rotor, and induction parameters
def test_get_flow_angle(bem_model):
    phi = bem_model.get_flow_angle(v0=10.0, r=10.0, a=0.3, a_prime=0.01, omega=2.0)
    assert isinstance(phi, float)
    assert 0 < phi < np.pi / 2

    phi_zero = bem_model.get_flow_angle(v0=10.0, r=10.0, a=0.0, a_prime=0.0, omega=2.0)
    phi_max = bem_model.get_flow_angle(v0=10.0, r=10.0, a=1.0, a_prime=1.0, omega=2.0)
    assert phi_zero > 0
    assert phi_max < np.pi / 2

# Test: Ensure local angle of attack is calculated correctly from flow angle and blade angles
def test_get_angle_attack(bem_model):
    phi = np.deg2rad(30)
    theta_p = 5.0
    twist = 2.0
    alpha = bem_model.get_angle_attack(phi, theta_p, twist)
    assert isinstance(alpha, float)
    assert -180 <= alpha <= 180

    phi_extreme = np.deg2rad(90)
    alpha_extreme = bem_model.get_angle_attack(phi_extreme, theta_p, twist)
    assert -180 <= alpha_extreme <= 180

# Test: Ensure normal and tangential force coefficients are correctly derived from Cl, Cd, and phi
def test_get_cn_ct(bem_model):
    cl, cd = 1.0, 0.01
    phi = np.deg2rad(30)
    cn, ct = bem_model.get_cn_ct(cl, cd, phi)
    assert isinstance(cn, float)
    assert isinstance(ct, float)

# Test: Validate that axial and tangential induction factors are updated and within physical range
def test_update_induction_factors(bem_model):
    phi = np.deg2rad(30)
    sigma = 0.05
    cn, ct = 1.0, 0.1
    a, a_prime = bem_model.update_induction_factors(phi, sigma, cn, ct)
    assert isinstance(a, float)
    assert isinstance(a_prime, float)
    assert 0 <= a < 1
    assert a_prime > 0

# Test: Verify local thrust and torque contributions are calculated and non-negative
def test_get_local_thrust_torque_contributions(bem_model):
    v0 = 10.0
    r = 10.0
    a = 0.3
    a_prime = 0.01
    omega = 2.0
    dT, dQ = bem_model.get_local_thrust_torque_contributions(v0, r, a, a_prime, omega)
    assert isinstance(dT, float)
    assert isinstance(dQ, float)
    assert dT >= 0
    assert dQ >= 0

# Test: Check that total thrust, torque, power, and coefficients are correctly integrated
def test_compute_thrust_torque(bem_model):
    omega = 2.0
    v0 = 10.0
    results = [{'d_thrust': 100.0, 'd_torque': 50.0, 'dr': 1.0} for _ in range(5)]
    t, q, p, ct, cp = bem_model.compute_thrust_torque(v0, results, omega)
    assert all(isinstance(val, float) for val in [t, q, p, ct, cp])
    assert t > 0 and q > 0 and p > 0
    assert 0 < ct < 2
    assert 0 < cp < 1

# Test: Ensure that chord, twist, and airfoil ID are returned correctly for a given radial position
def test_get_c_twist_afid(bem_model):
    r = 5.0
    chord, twist, afid = bem_model.get_c_twist_afid(r)

    print(f"Chord: {chord}, Twist: {twist}, Afid: {afid}")
    assert isinstance(chord, float), f"Expected float for chord, got {type(chord)}"
    assert isinstance(twist, float), f"Expected float for twist, got {type(twist)}"
    assert isinstance(afid, (float, np.float64)), f"Expected float for afid, got {type(afid)}"
    assert afid.is_integer(), f"Expected afid to be an integer-like value, got {afid}"
    afid = int(afid)
    assert isinstance(afid, int), f"Expected afid to be an integer, got {type(afid)}"
    assert afid >= 0, f"Expected non-negative afid, got {afid}"