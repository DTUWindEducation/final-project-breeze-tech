import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from data_io import BEMDataLoader

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
            tuple: (Cl, Cd) lift and drag coefficients
        """
        
        polar = next((p for p in self.polar_data if p['af_index'] == int(af_id)), None)
        
        # Create interpolation functions
        cl_interp = np.interp(alpha, polar['Alpha'], polar['Cl'])
        cd_interp = np.interp(alpha, polar['Alpha'], polar['Cd'])
        
        return cl_interp, cd_interp        

    def solve_bem_element(self, r, V0, theta_p, omega, a_init=0.0, a_prime_init=0.0, 
                         tol=1e-8, max_iter=1000):
        """
        Solve BEM equations for a single blade element.
        
        Args:
            r (float): Radial position of element (m)
            V0 (float): Wind speed (m/s)
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
            phi = np.arctan2((1 - a) * V0, (1 + a_prime) * omega * r)
            
            # Step 3: Compute angle of attack (convert to degrees)
            alpha = np.rad2deg(phi) - (theta_p + twist)
            
            # Step 4: Get Cl and Cd
            Cl, Cd = self.get_cl_cd(af_id, alpha)
                 
            # Step 5: Compute normal and tangential coefficients
            Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct = Cl * np.sin(phi) - Cd * np.cos(phi)
            
            # Step 6: Update induction factors
            denominator_a = 4 * np.sin(phi)**2 / (sigma * Cn)
            a = 1 / (denominator_a + 1)
            
            denominator_a_prime = 4 * np.sin(phi) * np.cos(phi) / (sigma * Ct) 
            a_prime = 1 / (denominator_a_prime - 1)
            
            # Check convergence
            if (abs(a - a_prev) < tol and abs(a_prime - a_prime_prev) < tol):
                break
        
        # Step 8: Compute thrust and torque contributions
        dT = 4 * np.pi * r * self.rho * V0**2 * a * (1 - a)
        dM = 4 * np.pi * r**3 * self.rho * V0 * omega * a_prime * (1 - a)
        
        return {
            'r': r,
            'a': a,
            'a_prime': a_prime,
            'phi': phi,
            'alpha': alpha,
            'Cl': Cl,
            'Cd': Cd,
            'Cn': Cn,
            'Ct': Ct,
            'dT': dT,
            'dM': dM
        }
    
    def compute_rotor_performance(self, V0, theta_p=None, omega=None):
        """
        Compute overall rotor performance for given operating conditions.
        
        Args:
            V0 (float): Wind speed (m/s)
                        
        Returns:
            dict: Performance metrics including power, thrust, torque, and coefficients
        """
        # Get operational parameters if not provided
        idx = np.argmin(np.abs(self.operational_data['wind_speed'] - V0))
        
        theta_p = self.operational_data.iloc[idx]['pitch']
        # Convert from rpm to rad/s
        omega = self.operational_data.iloc[idx]['rot_speed'] * (2*np.pi/60)
        
        # Get spans greater than 0
        r_elements = self.blade_data['BlSpn'].values
        r_elements = r_elements[r_elements > 0]  
        
        # Solve BEM for each element
        results = []
        for i in range(len(r_elements)):
            r = r_elements[i]
            elem_result = self.solve_bem_element(r, V0, theta_p, omega)
            # dr is the difference between the current and next span
            # unless it is the last element in which case its the last and end of rotor
            if i < (len(r_elements)-1):
                elem_result['dr'] = r_elements[i+1] - r_elements[i]
            else:
                elem_result['dr'] = self.blade_rad - r_elements[i]
                
            results.append(elem_result)
        
        # Integrate to get total thrust and torque
        total_thrust = sum(r['dT'] * r['dr'] for r in results)
        total_torque = sum(r['dM'] * r['dr'] for r in results)
        total_power = total_torque * omega
        
        # Compute coefficients
        A = np.pi * self.blade_rad**2  
        CT = total_thrust / (0.5 * self.rho * A * V0**2)
        CP = total_power / (0.5 * self.rho * A * V0**3)
        
        return {
            'V0': V0,
            'theta_p': theta_p,
            'omega': omega,
            'thrust': total_thrust,
            'torque': total_torque,
            'power': total_power,
            'CT': CT,
            'CP': CP,
            'element_results': results
        }
    
    
    def compute_power_curve(self):
        """
        Compute power and thrust curves over a range of wind speeds.
        
        Args:
            V0_range (tuple, optional): (min, max) wind speed range. If None, uses operational data range.
            n_points (int, optional): Number of points to evaluate. Default 20.
            
        Returns:
            DataFrame: Power curve data with columns for wind speed, power, thrust, etc.
        """
        # Determine wind speed range        
        V0_values = self.operational_data['wind_speed']
        
        # Compute performance at each wind speed
        results = []
        for V0 in V0_values:
            perf = self.compute_rotor_performance(V0)
            results.append({
                'V0': V0,
                'power': perf['power'],
                'thrust': perf['thrust'],
                'torque': perf['torque'],
                'CT': perf['CT'],
                'CP': perf['CP'],
                'pitch': perf['theta_p'],
                'omega': perf['omega']
            })
        
        return pd.DataFrame(results)
