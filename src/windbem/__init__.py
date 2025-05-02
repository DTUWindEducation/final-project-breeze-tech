import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .data_io import BEMDataLoader
from .compute import BEMTurbineModel


# Protection incase fomeone does from windbem import *
__all__ = ['BEMDataLoader', 'BEMTurbineModel']

def solve_bem_element(bem_model, r, v0, theta_p, omega, a_init=0.0, a_prime_init=0.0,
                         tol=1e-8, max_iter=1000):
    """
    Solve BEM equations for a single blade element.
        
    Args:
        r (float): Radial position of element (m)
        v0 (float): Wind speed (m/s)
        theta_p (float): Pitch angle (deg)
        omega (float): Rotational speed (rad/s)
        a_init (float, optional): Initial guess for axial induction. Default 0.
        a_prime_init (float, optional): Initial guess for tangential induction. Default 0.
        tol (float, optional): Convergence tolerance. Default 1e-8.
        max_iter (int, optional): Maximum iterations. Default 1000.
            
    Returns:
        dict: Solution containing induction factors, forces, and flow angles
    """
    # Local cord length, twist angle and airfoil id at this span position
    c, twist, af_id  = bem_model.get_c_twist_afid(r)

    # Local solidity
    sigma = bem_model.get_local_solidity(r, c)

    # Initialize induction factors
    a = a_init
    a_prime = a_prime_init

    for _ in range(max_iter):
        # Save old values for convergence check
        a_prev, a_prime_prev = a, a_prime

        # Flow angle (radians) and angle of attack (degrees)
        phi = bem_model.get_flow_angle(v0, r, a, a_prime, omega)
        alpha = bem_model.get_angle_attack(phi, theta_p, twist)

        # Lift coefficient and drag coefficient
        coef_lift, coef_drag = bem_model.get_cl_cd(af_id, alpha)

        # Force coefficients (normal and tangential)
        coef_normal, coef_tangent = bem_model.get_cn_ct(coef_lift, coef_drag, phi)

        # Update induction factors
        a, a_prime = bem_model.update_induction_factors(phi, sigma, coef_normal, coef_tangent)

        # Check convergence
        if (abs(a - a_prev) < tol and abs(a_prime - a_prime_prev) < tol):
            break

    # Local thrust and torque contributions
    d_thrust, d_torque = bem_model.get_local_thrust_torque_contributions(v0, r, a, a_prime, omega)

    return {
        'r': r,
        'a': a,
        'a_prime': a_prime,
        'phi': phi,
        'alpha': alpha,
        'coef_lift': coef_lift,
        'coef_drag': coef_drag,
        'coef_normal': coef_normal,
        'coef_tangent': coef_tangent,
        'd_thrust': d_thrust,
        'd_torque': d_torque
    }

def compute_rotor_performance(bem_model, v0, theta_p=None, omega=None):
    """
    Compute overall rotor performance for given operating conditions.

    Args:
        v0 (float): Wind speed (m/s).
        theta_p (float, optional): Pitch angle (deg). If None, uses operational data.
        omega (float, optional): Rotational speed (rad/s). If None, uses operational data.

    Returns:
        dict: Performance metrics including power, thrust, torque, and aerodynamic coefficients.
    """

    # Get operational parameters if not provided
    theta_p, omega = bem_model.get_operational_strategy(v0)

    # Element spans
    r_elements = bem_model.get_element_spans()

    # Solve BEM for each element
    results = []
    for i, r in enumerate(r_elements):
        elem_result = solve_bem_element(bem_model, r, v0, theta_p, omega)
        # dr is the difference between the current and next span
        # unless it is the last element in which case its the last and end of rotor
        if i < (len(r_elements)-1):
            elem_result['dr'] = r_elements[i+1] - r
        else:
            elem_result['dr'] = bem_model.blade_rad - r

        results.append(elem_result)

    # Integrate to get total thrust, torque, power and thrust & power coefficients
    total_thrust, total_torque, total_power, thrust_coef, power_coef = bem_model.compute_thrust_torque(v0, results, omega)

    return {
        'v0': v0,
        'theta_p': theta_p,
        'omega': omega,
        'thrust': total_thrust,
        'torque': total_torque,
        'power': total_power,
        'thrust_coef': thrust_coef,
        'power_coef': power_coef,
        'element_results': results
    }

def compute_power_curve(bem_model):
    """
    Compute power and thrust curves over a range of wind speeds.
        
    Returns:
        DataFrame: Power curve data with columns for wind speed, power, thrust, etc.
    """
    # Determine wind speed range
    v0_values = bem_model.operational_data['wind_speed']

    # Compute performance at each wind speed
    results = []
    for v0 in v0_values:
        perf = compute_rotor_performance(bem_model, v0)
        results.append({
            'v0': v0,
            'power': perf['power'],
            'thrust': perf['thrust'],
            'torque': perf['torque'],
            'thrust_coef': perf['thrust_coef'],
            'power_coef': perf['power_coef'],
            'pitch': perf['theta_p'],
            'omega': perf['omega']
        })

    return pd.DataFrame(results)

def plot_results(bem_model):
    power_curve = compute_power_curve(bem_model)
    true_thr = bem_model.operational_data["aero_thrust"]
    true_pow = bem_model.operational_data["aero_power"]
    true_wind = bem_model.operational_data["wind_speed"]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(power_curve['v0'], power_curve['power']/1e3, color = "blue", label = "BEM Power")
    plt.plot(true_wind, true_pow, color = "red", label = "True Power")
    plt.legend()
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power [kW]')
    plt.title('Power Curve')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(power_curve['v0'], power_curve['thrust']/1e3, color = "blue", label = "BEM Thrust")
    plt.plot(true_wind, true_thr, color = "red", label = "True Thrust")
    plt.legend()
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Thrust [kN]')
    plt.title('Thrust Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_results(performance):
    print(f"At {performance['v0']} m/s wind speed:")
    print(f"Power: {performance['power']/1e3:.2f} kW")
    print(f"Thrust: {performance['thrust']/1e3:.2f} kN")
    print(f"Power coefficient: {performance['power_coef']:.3f}")
