import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
    
    def __init__(self, blade_file, operational_file, polar_dir, rho=1.225, blade_rad=120, no_blades=3):
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
        self.blade_rad = blade_rad
        
        # Load operational strategy data
        self.operational_data = self._load_operational_data(operational_file)
        
        # Load airfoil polar data
        self.polar_data = self._load_polar_data(polar_dir)
        # Set constants
        self.rho = rho
        self.no_blades = no_blades  # Number of blades for IEA 15MW turbine
      
    def import_af_shapes(self, af_coords_file_path=None):
        """
        Imports coordinates of airfoil shapes from .txt files.
        Returns:
        - af_coords (list): List of normalized x-y coordinates of airfoil shapes [-].
        """
        if af_coords_file_path is None:
            af_coords_file_path = list(Path('inputs/IEA-15-240-RWT/Airfoils/').glob('IEA-15-240-RWT_AF*.*'))
        af_coords = []
        for file in af_coords_file_path:
            x, y = np.loadtxt(file, skiprows=8, unpack=True)
            af_coords.append([x, y])
        return af_coords

    def plot_af_shapes(self, af_coords=None):
        """
        Plots airfoil shapes in one figure.
        Parameters:
        - af_coords (list): List of normalized x-y coordinates of airfoil shapes [-].
        """
        if af_coords is None:
           af_coords = self.import_af_shapes()
        num = len(af_coords) 

        # Plot the data
        plt.figure(figsize=(8, 6))
        for i, data in enumerate(af_coords):
            x = data[0]
            y = data[1]
            plt.plot(x, y, marker='o', linestyle='-', markersize=4, label=f"Airfoil Shape {i+1}")

        # Formatting
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"Airfoil profiles (n={num})")
        #plt.legend()
        plt.grid(True)
        plt.axis("equal")

        plt.show()
        

    def _load_blade_data(self, aerodyn_file_path, skiprow_num=6, aerodyn_names = None):
        """Load blade geometry data from file."""   
        if aerodyn_names is None:
            aerodyn_names_ = ['BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist', 'BlChord', 
                            'BlAFID','BlCb', 'BlCenBn', 'BlCenBt']
        else:
            aerodyn_names_ = aerodyn_names
        
        aerodyn15_df = pd.read_csv(aerodyn_file_path, sep=r'\s+', skiprows=skiprow_num, 
                        names=aerodyn_names_)
        return aerodyn15_df
    
    
    
    def _load_operational_data(self, onshore_file_path, skiprow_num=1, onshore_names = None):
        """Load operational strategy data from file."""
        if onshore_names is None:
            onshore_names_ = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
        else:
            onshore_names_ = onshore_names
            
        onshore_15mw_df = pd.read_csv(onshore_file_path, sep=r'\s+', skiprows=skiprow_num, 
                            names=onshore_names_)
        return onshore_15mw_df
    

    def _load_polar_data(self, dir_path):
        """Load airfoil polar data from directory."""
        polar_files = list(Path(dir_path).glob('*Polar*.dat'))
        polar_data = []
        
        for file in polar_files:
            try:
                # Extract airfoil index from filename
                af_index = int(file.name.split('_')[-1].split('.')[0])
                # the file names start from 0 but they correspond to an af_index 1 higher
                af_index +=1
                
                # Load data
                data = pd.read_csv(file, sep=r'\s+', skiprows=54, 
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
