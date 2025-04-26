from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BEMDataLoader:
    """
    A class to implement Blade Element Momentum (BEM) theory for wind turbine performance analysis.
    
    This class models the aerodynamic performance of a wind turbine rotor using BEM theory,
    computing key performance metrics like power output, thrust, and torque.
    
    Attributes:
        blade_data (DataFrame): Blade geometry data (span, twist, chord, airfoil IDs)
        operational_data (DataFrame): Operational strategy data (wind speed vs pitch and RPM)
        polar_data (list): List of airfoil polar data (Cl, Cd vs alpha)
        rho (float): Air density (kg/m^3)
        blade_rad (float): Rotor radius (m)
        no_blades (int): Number of blades
    """

    def __init__(self, blade_file, operational_file, polar_dir, rho=1.225, blade_rad=120,
                 no_blades=3):
        """
        Initialize the BEM turbine model with input data.
        
        Args:
            blade_file (str): Path to blade geometry file
            operational_file (str): Path to operational strategy file
            polar_dir (str): Directory containing airfoil polar data files
            rho (float, optional): Air density in kg/m^3. Defaults to 1.225.
            blade_rad (float, optional): Rotor radius in meters. Default is 120.
            no_blades (int, optional): Number of blades. Default is 3.
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

        Args:
            af_coords_file_path (list, optional): List of file paths. If None, loads default.

        Returns:
            af_coords (list): List of normalized x-y coordinates of airfoil shapes [-].
        """
        default_path = 'inputs/IEA-15-240-RWT/Airfoils/'
        if af_coords_file_path is None:
            af_coords_file_path = list(Path(default_path).glob('IEA-15-240-RWT_AF*.*'))
        af_coords = []
        for file in af_coords_file_path:
            x, y = np.loadtxt(file, skiprows=8, unpack=True)
            af_coords.append([x, y])
        return af_coords

    def plot_af_shapes(self, af_coords=None):
        """
        Plots airfoil shapes in one figure.
        
        Args:
            af_coords (list, optional): List of [x, y] arrays. If None, imports default shapes.
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
        """
        Load blade geometry data from file.
        
        Args:
            aerodyn_file_path (str or Path): Path to the blade file.
            skiprow_num (int, optional): Number of top rows to skip in the file. Default is 6.
            aerodyn_names (list, optional): Custom column names. Default uses standard names.

        Returns:
            DataFrame: Blade geometry data.
        """

        if aerodyn_names is None:
            aerodyn_names_ = ['BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist', 'BlChord',
                            'BlAFID','BlCb', 'BlCenBn', 'BlCenBt']
        else:
            aerodyn_names_ = aerodyn_names

        aerodyn15_df = pd.read_csv(aerodyn_file_path, sep=r'\s+', skiprows=skiprow_num,
                        names=aerodyn_names_)
        return aerodyn15_df

    def _load_operational_data(self, onshore_file_path, skiprow_num=1, onshore_names = None):
        """
        Load operational strategy data from file.
        
        Args:
            onshore_file_path (str or Path): Path to the operational file.
            skiprow_num (int, optional): Number of top rows to skip in the file. Default is 1.
            onshore_names (list, optional): Custom column names. Default uses typical names.

        Returns:
            DataFrame: Operational data (wind speed, pitch, rotor speed, aero power, aero thrust).
        """
        if onshore_names is None:
            onshore_names_ = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
        else:
            onshore_names_ = onshore_names

        onshore_15mw_df = pd.read_csv(onshore_file_path, sep=r'\s+', skiprows=skiprow_num,
                            names=onshore_names_)
        return onshore_15mw_df

    def _load_polar_data(self, dir_path):
        """
        Load airfoil polar data from directory.

        Args:
            dir_path (str or Path): Path to the directory containing polar files.

        Returns:
            list: List of dicts of airfoil index and polar curve arrays (Alpha, Cl, Cd, Cm).
        """

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
