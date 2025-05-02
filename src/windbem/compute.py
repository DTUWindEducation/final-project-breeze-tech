"""Computing module containing the BEMTurbineModel class with methods necessary 
to calculate key parameters and final performance metrics such as 
power output, thrust, and torque."""
import numpy as np
from .data_io import BEMDataLoader

class BEMTurbineModel(BEMDataLoader):
    """
    A class to implement Blade Element Momentum (BEM) theory for wind turbine performance analysis.
    
    This class models the aerodynamic performance of a wind turbine rotor using BEM theory,
    computing key performance metrics like power output, thrust, and torque.
    """
    def get_operational_strategy(self, v0):
        """
        Compute optimal operational strategy - blade pitch angle and rotational speed
        for given wind speed.

        Args:
            v0 (float): Wind speed (m/s).
        
        Returns:
            tuple: (theta_p, omega) blade pitch angle (deg) and rotational speed (rad/s).
        """
        idx = np.argmin(np.abs(self.operational_data['wind_speed'] - v0))

        # Blade pitch angle
        theta_p = self.operational_data.iloc[idx]['pitch']
        # Rotational speed (converted from rpm to rad/s)
        omega = self.operational_data.iloc[idx]['rot_speed'] * (2*np.pi/60)

        return theta_p, omega

    def get_element_spans(self):
        """
        Get span positions of each element along the blade.
        
        Returns:
            ndarray: Span positions of blade elements.
        """
        r_elements = self.blade_data['BlSpn'].values
        r_elements = r_elements[r_elements > 0]

        return r_elements

    def get_c_twist_afid(self, r):
        """
        Get the local chord length, twist angle, and airfoil index for 
        the given element span position.

        Args:
            r (float): Radial position of blade element (m).

        Returns:
            tuple: (c, twist, af_id) local chord length (m), twist angle (deg), airfoil index.
        """
        idx = np.argmin(np.abs(self.blade_data['BlSpn'] - r))
        c = self.blade_data.iloc[idx]['BlChord']
        twist = self.blade_data.iloc[idx]['BlTwist']
        af_id = self.blade_data.iloc[idx]['BlAFID']

        return c, twist, af_id

    def get_local_solidity(self, r, c):
        """
        Get the local solidity for given span position and local chord length.

        Args:
            r (float): Radial position of blade element (m).
            c (float): Local chord length (m).

        Returns:
            float: Local solidity.
        """
        sigma = (self.no_blades * c) / (2 * np.pi * r)

        return sigma

    def get_flow_angle(self, v0, r, a, a_prime, omega):
        """
        Get flow angle for given wind speed, span position, induction factors and rotational speed.

        Args:
            v0 (float): Wind speed (m/s).
            r (float): Radial position of blade element (m).
            a (float): Axial induction factor.
            a_prime (float): Tangential induction factor.
            omega (float): Rotational speed (rad/s).

        Returns:
            float: Flow angle (rad)
        """
        phi = np.arctan2((1 - a) * v0, (1 + a_prime) * omega * r)

        return phi

    def get_angle_attack(self, phi, theta_p, twist):
        """
        Get the local angle of attack for given flow angle, blade pitch angle and twist angle.

        Args:
            phi (float): Flow angle (rad).
            theta_p (float): Blade pitch angle (deg).
            twist (float): Twist angle (deg).

        Returns:
            float: Local angle of attack (deg).
        """
        alpha = np.rad2deg(phi) - (theta_p + twist)

        return alpha

    def get_cl_cd(self, af_id, alpha):
        """
        Get lift and drag coefficients for given airfoil ID and angle of attack.
        
        Args:
            af_id (int): Airfoil ID (1-50 for IEA 15MW).
            alpha (float): Angle of attack in degrees.
            
        Returns:
            tuple: (coef_lift, coef_drag) lift and drag coefficients.
        """
        polar = next((p for p in self.polar_data if p['af_index'] == int(af_id)), None)

        # Create interpolation functions
        cl_interp = np.interp(alpha, polar['Alpha'], polar['Cl'])
        cd_interp = np.interp(alpha, polar['Alpha'], polar['Cd'])

        return cl_interp, cd_interp

    def get_cn_ct(self, coef_lift, coef_drag, phi):
        """
        Get normal and tangential force coefficients for given flow angle and 
        lift & drag coefficients.

        Args:
            coef_lift (float): Lift coefficient.
            coef_drag (float): Drag coefficient.
            phi (float): Flow angle (rad).

        Returns:
            tuple: (coef_normal, coef_tangent) normal and tangential force coefficients.
        """
        coef_normal = coef_lift * np.cos(phi) + coef_drag * np.sin(phi)
        coef_tangent = coef_lift * np.sin(phi) - coef_drag * np.cos(phi)

        return coef_normal, coef_tangent

    def update_induction_factors(self, phi, sigma, coef_normal, coef_tangent):
        """
        Update the induction factors with given flow angle, local solidity, and force coefficients.

        Args:
            phi (float): Flow angle (rad).
            sigma (float): local solidity.
            coef_normal (float): normal force coefficient.
            coef_tangent (float): tangential force coefficient.

        Returns:
            tuple: (a, a_prime) axial and tangential induction factors.
        """
        denominator_a = 4 * np.sin(phi)**2 / (sigma * coef_normal)
        a = 1 / (denominator_a + 1)

        denominator_a_prime = 4 * np.sin(phi) * np.cos(phi) / (sigma * coef_tangent)
        a_prime = 1 / (denominator_a_prime - 1)

        return a, a_prime

    def get_local_thrust_torque_contributions(self, v0, r, a, a_prime, omega):
        """
        Get the local thrust and torque contributions for given wind speed, span position,
        induction factors and rotational speed.

        Args:
            v0 (float): Wind speed (m/s).
            r (float): Radial position of blade element (m).
            a (float): Axial induction factor.
            a_prime (float): Tangential induction factor.
            omega (float): Rotational speed (rad/s).

        Returns:
            tuple: (d_thrust, d_torque) local thrust (N) and torque (Nm) contributions.
        """
        d_thrust = 4 * np.pi * r * self.rho * v0**2 * a * (1 - a)
        d_torque = 4 * np.pi * r**3 * self.rho * v0 * omega * a_prime * (1 - a)

        return d_thrust, d_torque

    def compute_thrust_torque(self, v0, results, omega):
        """
        Compute the total thrust, torque and power along with coefficients of thrust and power,
        based on given wind speed, elements results and rotational speed.
        Elements of 'results' list (dictionaries) must contain 'd_thrust' and 'd_torque' keys.

        Args:
            v0 (float): Wind speed (m/s).
            results (list): List of results for each blade element, containing local thrust and 
                torque contributions.
            omega (float): Rotational speed (rad/s).

        Returns:
            tuple: (total_thrust, total_torque, total_power, thrust_coef, power_coef)
                total thrust (N), torque (Nm), power (W) and coefficients of thrust and power.
        """
        # Integrate to get total thrust and torque
        total_thrust = sum(r['d_thrust'] * r['dr'] for r in results)
        total_torque = sum(r['d_torque'] * r['dr'] for r in results)
        total_power = total_torque * omega

        # Compute coefficients
        area_rotor = np.pi * self.blade_rad**2
        thrust_coef = total_thrust / (0.5 * self.rho * area_rotor * v0**2)
        power_coef = total_power / (0.5 * self.rho * area_rotor * v0**3)

        return total_thrust, total_torque, total_power, thrust_coef, power_coef
