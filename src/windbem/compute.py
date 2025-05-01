import numpy as np
import pandas as pd

from .data_io import BEMDataLoader

class BEMTurbineModel(BEMDataLoader):
    """
    A class to implement Blade Element Momentum (BEM) theory for wind turbine performance analysis.
    
    This class models the aerodynamic performance of a wind turbine rotor using BEM theory,
    computing key performance metrics like power output, thrust, and torque.
    """
    def get_operational_strategy(self, v0):
        idx = np.argmin(np.abs(self.operational_data['wind_speed'] - v0))

        # Blade pitch angle
        theta_p = self.operational_data.iloc[idx]['pitch']
        # Rotational speed (converted from rpm to rad/s)
        omega = self.operational_data.iloc[idx]['rot_speed'] * (2*np.pi/60)
        return theta_p, omega
   
    def get_element_spans(self):
        r_elements = self.blade_data['BlSpn'].values
        r_elements = r_elements[r_elements > 0]
        return r_elements

    def get_c_twist_afid(self, r):
        idx = np.argmin(np.abs(self.blade_data['BlSpn'] - r))
        c = self.blade_data.iloc[idx]['BlChord']
        twist = self.blade_data.iloc[idx]['BlTwist']
        af_id = self.blade_data.iloc[idx]['BlAFID']
        return c, twist, af_id

    def get_local_solidity(self, r, c):
        sigma = (self.no_blades * c) / (2 * np.pi * r)

        return sigma

    def get_flow_angle(self, v0, r, a, a_prime, omega):
        phi = np.arctan2((1 - a) * v0, (1 + a_prime) * omega * r)

        return phi
    
    def get_angle_attack(self, phi, theta_p, twist):
        alpha = np.rad2deg(phi) - (theta_p + twist)

        return alpha

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
    
    def get_cn_ct(self, coef_lift, coef_drag, phi):
        coef_normal = coef_lift * np.cos(phi) + coef_drag * np.sin(phi)
        coef_tangent = coef_lift * np.sin(phi) - coef_drag * np.cos(phi)

        return coef_normal, coef_tangent

    def update_induction_factors(self, phi, sigma, coef_normal, coef_tangent):
        denominator_a = 4 * np.sin(phi)**2 / (sigma * coef_normal)
        a = 1 / (denominator_a + 1)

        denominator_a_prime = 4 * np.sin(phi) * np.cos(phi) / (sigma * coef_tangent)
        a_prime = 1 / (denominator_a_prime - 1)

        return a, a_prime
    
    def get_local_thrust_torque_contributions(self, v0, r, a, a_prime, omega):
        d_thrust = 4 * np.pi * r * self.rho * v0**2 * a * (1 - a)
        d_torque = 4 * np.pi * r**3 * self.rho * v0 * omega * a_prime * (1 - a)

        return d_thrust, d_torque

    def compute_thrust_torque(self, v0, results, omega):
        # Integrate to get total thrust and torque
        total_thrust = sum(r['d_thrust'] * r['dr'] for r in results)
        total_torque = sum(r['d_torque'] * r['dr'] for r in results)
        total_power = total_torque * omega

        # Compute coefficients
        area_rotor = np.pi * self.blade_rad**2
        thrust_coef = total_thrust / (0.5 * self.rho * area_rotor * v0**2)
        power_coef = total_power / (0.5 * self.rho * area_rotor * v0**3)

        return total_thrust, total_torque, total_power, thrust_coef, power_coef
