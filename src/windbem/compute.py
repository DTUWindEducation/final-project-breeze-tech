import numpy as np
import pandas as pd
#from scipy.interpolate import interp1d
from .data_io import BEMDataLoader

class BEMTurbineModel(BEMDataLoader):
    """
    A class to implement Blade Element Momentum (BEM) theory for wind turbine performance analysis.
    
    This class models the aerodynamic performance of a wind turbine rotor using BEM theory,
    computing key performance metrics like power output, thrust, and torque.
    """

    def get_cl_cd(self, af_id, alpha):
        """
        Get lift and drag coefficients for given airfoil ID and angle of attack.
        
        Args:
            af_id (int): Airfoil ID (1-50 for IEA 15MW)
            alpha (float): Angle of attack in degrees
            
        Returns:
            tuple: (coef_lift, coef_drag) lift and drag coefficients
        """

        polar = next((p for p in self.polar_data if p['af_index'] == int(af_id)), None)

        # Create interpolation functions
        cl_interp = np.interp(alpha, polar['Alpha'], polar['Cl'])
        cd_interp = np.interp(alpha, polar['Alpha'], polar['Cd'])

        return cl_interp, cd_interp

    def solve_bem_element(self, r, v0, theta_p, omega, a_init=0.0, a_prime_init=0.0,
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
        # Get blade geometry at this radial position
        idx = np.argmin(np.abs(self.blade_data['BlSpn'] - r))

        # Get blade geometry at this radial position
        c = self.blade_data.iloc[idx]['BlChord']
        twist = self.blade_data.iloc[idx]['BlTwist']
        af_id = self.blade_data.iloc[idx]['BlAFID']

        # Local solidity
        sigma = (self.no_blades * c) / (2 * np.pi * r)

        # Initialize induction factors
        a = a_init
        a_prime = a_prime_init

        for _ in range(max_iter):
            # Save old values for convergence check
            a_prev, a_prime_prev = a, a_prime

            # Step 2: Compute flow angle (radians)
            phi = np.arctan2((1 - a) * v0, (1 + a_prime) * omega * r)

            # Step 3: Compute angle of attack (convert to degrees)
            alpha = np.rad2deg(phi) - (theta_p + twist)

            # Step 4: Get coef_lift and coef_drag
            coef_lift, coef_drag = self.get_cl_cd(af_id, alpha)

            # Step 5: Compute normal and tangential coefficients
            coef_normal = coef_lift * np.cos(phi) + coef_drag * np.sin(phi)
            coef_tangent = coef_lift * np.sin(phi) - coef_drag * np.cos(phi)

            # Step 6: Update induction factors
            denominator_a = 4 * np.sin(phi)**2 / (sigma * coef_normal)
            a = 1 / (denominator_a + 1)

            denominator_a_prime = 4 * np.sin(phi) * np.cos(phi) / (sigma * coef_tangent)
            a_prime = 1 / (denominator_a_prime - 1)

            # Check convergence
            if (abs(a - a_prev) < tol and abs(a_prime - a_prime_prev) < tol):
                break

        # Step 8: Compute thrust and torque contributions
        d_thrust = 4 * np.pi * r * self.rho * v0**2 * a * (1 - a)
        d_torque = 4 * np.pi * r**3 * self.rho * v0 * omega * a_prime * (1 - a)

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

    def compute_rotor_performance(self, v0, theta_p=None, omega=None):
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
        idx = np.argmin(np.abs(self.operational_data['wind_speed'] - v0))

        theta_p = self.operational_data.iloc[idx]['pitch']
        # Convert from rpm to rad/s
        omega = self.operational_data.iloc[idx]['rot_speed'] * (2*np.pi/60)

        # Get spans greater than 0
        r_elements = self.blade_data['BlSpn'].values
        r_elements = r_elements[r_elements > 0]

        # Solve BEM for each element
        results = []
        for i, r_elem in enumerate(r_elements):
            r = r_elements[i]
            elem_result = self.solve_bem_element(r, v0, theta_p, omega)
            # dr is the difference between the current and next span
            # unless it is the last element in which case its the last and end of rotor
            if i < (len(r_elements)-1):
                elem_result['dr'] = r_elements[i+1] - r_elem
            else:
                elem_result['dr'] = self.blade_rad - r_elem

            results.append(elem_result)

        # Integrate to get total thrust and torque
        total_thrust = sum(r['d_thrust'] * r['dr'] for r in results)
        total_torque = sum(r['d_torque'] * r['dr'] for r in results)
        total_power = total_torque * omega

        # Compute coefficients
        area_rotor = np.pi * self.blade_rad**2
        thrust_coef = total_thrust / (0.5 * self.rho * area_rotor * v0**2)
        power_coef = total_power / (0.5 * self.rho * area_rotor * v0**3)

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


    def compute_power_curve(self):
        """
        Compute power and thrust curves over a range of wind speeds.
        
        Returns:
            DataFrame: Power curve data with columns for wind speed, power, thrust, etc.
        """
        # Determine wind speed range
        v0_values = self.operational_data['wind_speed']

        # Compute performance at each wind speed
        results = []
        for v0 in v0_values:
            perf = self.compute_rotor_performance(v0)
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
