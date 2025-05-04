#--------------- import if package not installed ---------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
#---------------------------------------------------------------

from windbem import BEMTurbineModel
from windbem import plot_results
from windbem import print_results
import matplotlib.pyplot as plt
import numpy as np

def main():
    V0 = 10  # m/s
    # 1. Load and parse the provided turbine data.
    bem_model = BEMTurbineModel(
        blade_file ='inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat',
        operational_file ='inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt',
        polar_dir ='inputs/IEA-15-240-RWT/Airfoils/'
        )

    # 2. Plot the provided airfoil shapes in one figure.
    bem_model.plot_af_shapes()

    # 6. Compute optimal operational strategy i.e., blade pitch angle and
    # rotational speed, as function of wind speed based on the provided operational strategy.
    theta_p, omega = bem_model.get_operational_strategy(V0)

    # Element spans
    r_elements = bem_model.get_element_spans()

    # Solve BEM for each element
    results = []
    for i, r in enumerate(r_elements):
        # Local cord length, twist angle and airfoil id at this span position
        c, twist, af_id  = bem_model.get_c_twist_afid(r)

        # Local solidity
        sigma = bem_model.get_local_solidity(r, c)

        # Initialize induction factors (Default 0)
        a = 0.0
        a_prime = 0.0
        tol = 1e-8 # Convergence tolerance. Default 1e-8.

        max_iter = 1000 # Maximum iterations. Default 1000.
        for _ in range(max_iter):
            # Save old values for convergence check
            a_prev, a_prime_prev = a, a_prime

            # Flow angle (radians) and angle of attack (degrees)
            phi = bem_model.get_flow_angle(V0, r, a, a_prime, omega)
            alpha = bem_model.get_angle_attack(phi, theta_p, twist)

            # 3. Compute lift coefficient and drag coefficient as function of span position and
            # angle of attack.
            coef_lift, coef_drag = bem_model.get_cl_cd(af_id, alpha)

            # Force coefficients (normal and tangential)
            coef_normal, coef_tangent = bem_model.get_cn_ct(coef_lift, coef_drag, phi)

            # 4. Compute (update) the axial and tangential induction factors as function of span 
            # position, the inflow wind speed, the blade pitch angle and the rotational speed.
            a, a_prime = bem_model.update_induction_factors(phi, sigma, coef_normal, coef_tangent)

            # Check convergence
            if (abs(a - a_prev) < tol and abs(a_prime - a_prime_prev) < tol):
                break

        # Local thrust and torque contributions
        d_thrust, d_torque = bem_model.get_local_thrust_torque_contributions(V0, r, a, a_prime, omega)

        elem_result = {
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

        # dr is the difference between the current and next span
        # unless it is the last element, in which case it's the last, and end of the rotor.
        if i < (len(r_elements)-1):
            elem_result['dr'] = r_elements[i+1] - r
        else:
            elem_result['dr'] = bem_model.blade_rad - r

        results.append(elem_result)

    # 5. Compute the thrust, torque, and power of the rotor as function of the inflow
    # wind speed, the blade pitch angle and the rotational speed.
    total_thrust, total_torque, total_power, thrust_coef, power_coef = bem_model.compute_thrust_torque(V0, results, omega)
    
    # End results
    performance = {
        'v0': V0,
        'theta_p': theta_p,
        'omega': omega,
        'thrust': total_thrust,
        'torque': total_torque,
        'power': total_power,
        'thrust_coef': thrust_coef,
        'power_coef': power_coef,
        'element_results': results
    }

    # 7. Compute and plot power curve and thrust curve based on the 
    # optimal operational strategy obtained in the previous function.
    plot_results(bem_model)

    # EXTRA FUNCTION: Printing results
    print_results(performance)

if __name__ == "__main__":
    main()
