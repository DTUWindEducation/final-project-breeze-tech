import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

class BEMTurbineModel:
    """
    A class to implement Blade Element Momentum (BEM) theory for wind turbine performance analysis.
    
    This class models the aerodynamic performance of a wind turbine rotor using BEM theory,
    computing key performance metrics like power output, thrust, and torque.
    
    Attributes:
        blade_data (DataFrame): Blade geometry data (span, twist, chord, airfoil IDs)
        operational_data (DataFrame): Operational strategy data (wind speed vs pitch and RPM)
        polar_data (list): List of airfoil polar data (Cl, Cd vs alpha)
        rho (float): Air density (kg/m^3)
        R (float): Rotor radius (m)
        B (int): Number of blades
    """
    
    def __init__(self, blade_file, operational_file, polar_dir, rho=1.225):
        """
        Initialize the BEM turbine model with input data.
        
        Args:
            blade_file (str): Path to blade geometry file
            operational_file (str): Path to operational strategy file
            polar_dir (str): Directory containing airfoil polar data files
            rho (float, optional): Air density in kg/m^3. Defaults to 1.225.
        """
        # Load blade geometry data
        self.blade_data = self._load_blade_data(blade_file)
        self.R = self.blade_data['BlSpn'].max()  # Rotor radius
        
        # Load operational strategy data
        self.operational_data = self._load_operational_data(operational_file)
        
        # Load airfoil polar data
        self.polar_data = self._load_polar_data(polar_dir)
        
        # Set constants
        self.rho = rho
        self.B = 3  # Number of blades for IEA 15MW turbine
        
    def _load_blade_data(self, file_path):
        """Load blade geometry data from file."""
        column_names = ['BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist', 
                       'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt']
        return pd.read_csv(file_path, sep=r'\s+', skiprows=6, names=column_names)
    
    def _load_operational_data(self, file_path):
        """Load operational strategy data from file."""
        column_names = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
        return pd.read_csv(file_path, sep=r'\s+', skiprows=1, names=column_names)
    
    def _load_polar_data(self, dir_path):
        """Load airfoil polar data from directory."""
        polar_files = list(Path(dir_path).glob('*Polar*.dat'))
        polar_data = []
        
        for file in polar_files:
            try:
                # Extract airfoil index from filename
                af_index = int(file.name.split('_')[-1].split('.')[0])
                
                # Load data
                data = pd.read_csv(file, sep=r'\s+', comment='!', 
                                 names=['Alpha', 'Cl', 'Cd', 'Cm'])
                
                polar_data.append({
                    'af_index': af_index,
                    'Alpha': data['Alpha'].values,
                    'Cl': data['Cl'].values,
                    'Cd': data['Cd'].values,
                    'Cm': data['Cm'].values
                })
            except (ValueError, IndexError):
                continue
                
        return sorted(polar_data, key=lambda x: x['af_index'])
    
    def get_cl_cd(self, af_id, alpha):
        """
        Get lift and drag coefficients for given airfoil ID and angle of attack.
        
        Args:
            af_id (int): Airfoil ID (1-50 for IEA 15MW)
            alpha (float): Angle of attack in degrees
            
        Returns:
            tuple: (Cl, Cd) lift and drag coefficients
        """
        # Find the polar data for this airfoil
        polar = next((p for p in self.polar_data if p['af_index'] == af_id), None)
        if polar is None:
            raise ValueError(f"Airfoil ID {af_id} not found in polar data")
        
        # Create interpolation functions
        cl_interp = interp1d(polar['Alpha'], polar['Cl'], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
        cd_interp = interp1d(polar['Alpha'], polar['Cd'], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        
        return float(cl_interp(alpha)), float(cd_interp(alpha))
    
    def solve_bem_element(self, r, V0, theta_p, omega, a_init=0.0, a_prime_init=0.0, 
                         tol=1e-6, max_iter=100):
        """
        Solve BEM equations for a single blade element.
        
        Args:
            r (float): Radial position of element (m)
            V0 (float): Wind speed (m/s)
            theta_p (float): Pitch angle (deg)
            omega (float): Rotational speed (rad/s)
            a_init (float, optional): Initial guess for axial induction. Default 0.
            a_prime_init (float, optional): Initial guess for tangential induction. Default 0.
            tol (float, optional): Convergence tolerance. Default 1e-6.
            max_iter (int, optional): Maximum iterations. Default 100.
            
        Returns:
            dict: Solution containing induction factors, forces, and flow angles
        """
        # Get blade geometry at this radial position
        blade_props = self._get_blade_properties(r)
        c = blade_props['BlChord']
        twist = blade_props['BlTwist']
        af_id = blade_props['BlAFID']
        
        # Local solidity
        sigma = (self.B * c) / (2 * np.pi * r)
        
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
            try:
                Cl, Cd = self.get_cl_cd(af_id, alpha)
            except ValueError:
                Cl, Cd = 0.0, 0.0  # Handle cases where alpha is out of polar range
            
            # Step 5: Compute normal and tangential coefficients
            Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct = Cl * np.sin(phi) - Cd * np.cos(phi)
            
            # Step 6: Update induction factors
            denominator_a = 4 * np.sin(phi)**2 / (sigma * Cn) if sigma * Cn != 0 else 1e10
            a = 1 / (denominator_a + 1)
            
            denominator_a_prime = 4 * np.sin(phi) * np.cos(phi) / (sigma * Ct) if sigma * Ct != 0 else 1e10
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
    
    def _get_blade_properties(self, r):
        """Get blade properties at given radial position using linear interpolation."""
        # Find the two closest radial stations
        idx = np.searchsorted(self.blade_data['BlSpn'].values, r)
        
        if idx == 0:
            return self.blade_data.iloc[0].to_dict()
        elif idx == len(self.blade_data):
            return self.blade_data.iloc[-1].to_dict()
        
        # Interpolate between the two closest stations
        r1 = self.blade_data.iloc[idx-1]
        r2 = self.blade_data.iloc[idx]
        
        # Linear interpolation factor
        f = (r - r1['BlSpn']) / (r2['BlSpn'] - r1['BlSpn'])
        
        # Interpolate all properties
        props = {}
        for col in self.blade_data.columns:
            if col == 'BlAFID':  # Airfoil ID - round to nearest integer
                props[col] = round(r1[col] + f * (r2[col] - r1[col]))
            else:
                props[col] = r1[col] + f * (r2[col] - r1[col])
        
        return props
    
    def compute_rotor_performance(self, V0, theta_p=None, omega=None):
        """
        Compute overall rotor performance for given operating conditions.
        
        Args:
            V0 (float): Wind speed (m/s)
            theta_p (float, optional): Pitch angle (deg). If None, uses optimal from operational data.
            omega (float, optional): Rotational speed (rad/s). If None, uses optimal from operational data.
            
        Returns:
            dict: Performance metrics including power, thrust, torque, and coefficients
        """
        # Get operational parameters if not provided
        if theta_p is None or omega is None:
            op_params = self.get_optimal_operational_params(V0)
            theta_p = theta_p if theta_p is not None else op_params['pitch']
            omega = omega if omega is not None else np.deg2rad(op_params['rot_speed'] * (2*np.pi/60))
        
        # Discretize blade into elements (using blade data stations)
        r_stations = self.blade_data['BlSpn'].values
        dr = np.diff(r_stations)
        r_elements = r_stations[:-1] + dr/2  # Midpoints between stations
        
        # Solve BEM for each element
        results = []
        for r, delta_r in zip(r_elements, dr):
            elem_result = self.solve_bem_element(r, V0, theta_p, omega)
            elem_result['dr'] = delta_r
            results.append(elem_result)
        
        # Integrate to get total thrust and torque
        total_thrust = sum(r['dT'] * r['dr'] for r in results)
        total_torque = sum(r['dM'] * r['dr'] for r in results)
        total_power = total_torque * omega
        
        # Compute coefficients
        A = np.pi * self.R**2  # Rotor area
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
    
    def get_optimal_operational_params(self, V0):
        """
        Get optimal pitch angle and rotational speed for given wind speed.
        
        Args:
            V0 (float): Wind speed (m/s)
            
        Returns:
            dict: Optimal pitch angle (deg) and rotational speed (rpm)
        """
        # Find closest wind speed in operational data
        idx = np.argmin(np.abs(self.operational_data['wind_speed'] - V0))
        return {
            'pitch': self.operational_data.iloc[idx]['pitch'],
            'rot_speed': self.operational_data.iloc[idx]['rot_speed']
        }
    
    def compute_power_curve(self, V0_range=None, n_points=20):
        """
        Compute power and thrust curves over a range of wind speeds.
        
        Args:
            V0_range (tuple, optional): (min, max) wind speed range. If None, uses operational data range.
            n_points (int, optional): Number of points to evaluate. Default 20.
            
        Returns:
            DataFrame: Power curve data with columns for wind speed, power, thrust, etc.
        """
        # Determine wind speed range
        if V0_range is None:
            V0_min = self.operational_data['wind_speed'].min()
            V0_max = self.operational_data['wind_speed'].max()
        else:
            V0_min, V0_max = V0_range
        
        V0_values = np.linspace(V0_min, V0_max, n_points)
        
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
