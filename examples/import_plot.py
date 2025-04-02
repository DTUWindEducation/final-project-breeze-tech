# wk 8 importing and plotting homework
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# there aare 4 types of files 
# - IEA-15-240-RWT_AeroDyn15_blade.dat
# - IEA_15MW_RWT_Onshore.opt
# - Airfoils/IEA-15-240-RWT_AF00_Coords.txt
# - IEA-15-240-RWT_AeroDyn15_Polar_00.dat

# Allocations:
# Finn - IEA_15MW_RWT_Onshore.opt, IEA-15-240-RWT_AeroDyn15_blade.dat

#########################################
# Run this from the main directory!
#########################################

onshore_15mw_file_path = 'inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt'
aerodyn15_file_path = 'inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat'

onshore_names = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
onshore_15mw_df = pd.read_csv(onshore_15mw_file_path, sep=r'\s+', skiprows=1, 
                    names=onshore_names)
#print(onshore_15mw_df.head())
plt.plot(onshore_15mw_df['wind_speed'], onshore_15mw_df['aero_thrust'])
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Aero Thrust (kn)')
plt.title('Wind Speed vs Aero Thrust')
plt.show()

aerodyn_names = ['BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist', 'BlChord', 'BlAFID',
                 'BlCb', 'BlCenBn', 'BlCenBt']
aerodyn15_df = pd.read_csv(aerodyn15_file_path, sep=r'\s+', skiprows=6, 
                    names=aerodyn_names)
#print(aerodyn15_df.head())
plt.plot(aerodyn15_df['BlSpn'], aerodyn15_df['BlCrvAC'])
plt.xlabel('BlSpn (m)')
plt.ylabel('BlCrvAC (deg)')
plt.title('BlSpn vs BlCrvAC')
plt.show()

# Hubert - Airfoils/IEA-15-240-RWT_AF00_Coords.txt

def import_af_shapes():
    """
    Imports coordinates of airfoil shapes from .txt files.
    Returns:
    - af_coords (list): List of normalized x-y coordinates of airfoil shapes [-].
    """
    af_coords_file_path = list(Path('inputs/IEA-15-240-RWT/Airfoils/').glob('IEA-15-240-RWT_AF*.*'))
    af_coords = []
    for file in af_coords_file_path:
        x, y = np.loadtxt(file, skiprows=8, unpack=True)
        af_coords.append([x, y])
    return af_coords

def plot_af_shapes(af_coords):
    """
    Plots airfoil shapes in one figure.
    Parameters:
    - af_coords (list): List of normalized x-y coordinates of airfoil shapes [-].
    """
    num = len(af_coords) # Number of shapes

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

af_coordinates = import_af_shapes()
plot_af_shapes(af_coordinates)

# Irene - IEA-15-240-RWT_AeroDyn15_Polar_00.dat

def import_polar_data():
    """
    Imports aerodynamic coefficients from polar data files.
    
    Returns:
    - polar_data (list): List of dictionaries containing aerodynamic data for each airfoil.
                        Each dictionary contains 'Alpha', 'Cl', 'Cd', and 'Cm' arrays.
    """
    # Define the directory where polar data files are located
    polar_dir = Path('inputs/IEA-15-240-RWT/Airfoils/')
    
    # Get the list of polar data files with a specific naming pattern
    polar_files_path = list(polar_dir.glob('*Polar*.dat'))
    
    # If no files are found, try another naming pattern
    if len(polar_files_path) == 0:
        polar_files_path = list(polar_dir.glob('*_AeroDyn15_Polar_*.dat'))
    
    polar_data = []
    
    for file in polar_files_path:
        # Extract the aerodynamic profile index from the filename
        file_name = file.name
        try:
            af_index = int(file_name.split('_')[-1].split('.')[0])
        except ValueError:
            continue
        
        try:
            # Attempt to load data using numpy
            data_array = None
            with open(file, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip() and not line.strip().startswith('!'):
                        try:
                            values = line.strip().split()
                            if len(values) >= 4:  # Ensure there are at least 4 columns (Alpha, Cl, Cd, Cm)
                                float(values[0])  # Check if the first four values are numeric
                                float(values[1])
                                float(values[2])
                                float(values[3])
                                data_array = np.loadtxt(file, skiprows=i)  # Load data from this row onwards
                                break
                        except ValueError:
                            continue
            
            # If numpy failed, manually parse the data
            if data_array is None:
                alpha, cl, cd, cm = [], [], [], []
                with open(file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.strip().startswith('!'):
                            try:
                                values = line.strip().split()
                                if len(values) >= 4:
                                    alpha.append(float(values[0]))
                                    cl.append(float(values[1]))
                                    cd.append(float(values[2]))
                                    cm.append(float(values[3]))
                            except ValueError:
                                continue
                if len(alpha) == len(cl) == len(cd) == len(cm) and len(alpha) > 0:
                    polar_data.append({'af_index': af_index, 'Alpha': np.array(alpha), 'Cl': np.array(cl), 'Cd': np.array(cd), 'Cm': np.array(cm)})
            else:
                # Ensure the data has at least 4 columns
                if data_array.shape[1] >= 4:
                    polar_data.append({'af_index': af_index, 'Alpha': data_array[:, 0], 'Cl': data_array[:, 1], 'Cd': data_array[:, 2], 'Cm': data_array[:, 3]})
        
        except Exception:
            continue
    
    # Sort the data by aerodynamic profile index
    polar_data.sort(key=lambda x: x['af_index'])
    return polar_data

def plot_polar_data(polar_data, plot_type='cl_alpha'):
    """
    Generates plots of aerodynamic coefficients.
    
    Parameters:
    - polar_data (list): List of dictionaries containing aerodynamic data.
    - plot_type (str): Type of plot ('cl_alpha', 'cd_alpha', 'cl_cd', 'cm_alpha').
    """
    if not polar_data:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Select a subset of profiles to plot if there are too many
    indices_to_plot = range(0, len(polar_data), max(1, len(polar_data)//10))
    
    for i in indices_to_plot:
        data = polar_data[i]
        af_index = data['af_index']
        
        # Plot Cl vs Alpha
        if plot_type == 'cl_alpha':
            plt.plot(data['Alpha'], data['Cl'], label=f'AF{af_index:02d}')
            plt.xlabel('Angle of Attack, α [deg]')
            plt.ylabel('Lift Coefficient, Cl [-]')
            plt.title('Lift Coefficient vs Angle of Attack')
        # Plot Cd vs Alpha
        elif plot_type == 'cd_alpha':
            plt.plot(data['Alpha'], data['Cd'], label=f'AF{af_index:02d}')
            plt.xlabel('Angle of Attack, α [deg]')
            plt.ylabel('Drag Coefficient, Cd [-]')
            plt.title('Drag Coefficient vs Angle of Attack')
        # Plot Cm vs Alpha
        elif plot_type == 'cm_alpha':
            plt.plot(data['Alpha'], data['Cm'], label=f'AF{af_index:02d}')
            plt.xlabel('Angle of Attack, α [deg]')
            plt.ylabel('Moment Coefficient, Cm [-]')
            plt.title('Moment Coefficient vs Angle of Attack')            
        # Plot Cl vs Cd (polar diagram)
        elif plot_type == 'cl_cd':
            plt.plot(data['Cd'], data['Cl'], label=f'AF{af_index:02d}')
            plt.xlabel('Drag Coefficient, Cd [-]')
            plt.ylabel('Lift Coefficient, Cl [-]')
            plt.title('Lift vs Drag Coefficient (Polar Diagram)')

    
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plot_type}_plot.png')  # Save the plot
    plt.show()

# Main execution block
if __name__ == "__main__":
    polar_data = import_polar_data()
    
    if polar_data:
        plot_polar_data(polar_data, plot_type='cl_alpha')
        plot_polar_data(polar_data, plot_type='cd_alpha')
        plot_polar_data(polar_data, plot_type='cm_alpha')
        plot_polar_data(polar_data, plot_type='cl_cd')


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def import_polar_data():
    """
    Imports aerodynamic coefficients from polar data files.
    
    Returns:
    - polar_data (list): List of dictionaries containing aerodynamic data for each airfoil.
                        Each dictionary contains 'Alpha', 'Cl', 'Cd', and 'Cm' arrays.
    """
    polar_dir = Path('inputs/IEA-15-240-RWT/Airfoils/')
    polar_files_path = list(polar_dir.glob('*Polar*.dat')) or list(polar_dir.glob('*_AeroDyn15_Polar_*.dat'))
    polar_data = []
    
    for file in polar_files_path:
        try:
            af_index = int(file.stem.split('_')[-1])
        except ValueError:
            continue
        
        try:
            data_array = np.genfromtxt(file, comments='!', skip_header=0)
            if data_array.shape[1] >= 4:
                polar_data.append({'af_index': af_index, 'Alpha': data_array[:, 0], 'Cl': data_array[:, 1], 'Cd': data_array[:, 2], 'Cm': data_array[:, 3]})
        except Exception:
            continue
    
    polar_data.sort(key=lambda x: x['af_index'])
    return polar_data

def plot_polar_data(polar_data, plot_type='cl_alpha'):
    """
    Generates plots of aerodynamic coefficients.
    
    Parameters:
    - polar_data (list): List of dictionaries containing aerodynamic data.
    - plot_type (str): Type of plot ('cl_alpha', 'cd_alpha', 'cl_cd', 'cm_alpha').
    """
    if not polar_data:
        return
    
    plt.figure(figsize=(10, 8))
    indices_to_plot = range(0, len(polar_data), max(1, len(polar_data)//10))
    
    for i in indices_to_plot:
        data = polar_data[i]
        af_index = data['af_index']
        
        plot_options = {
            'cl_alpha': ('Alpha', 'Cl', 'Angle of Attack, α [deg]', 'Lift Coefficient, Cl [-]', 'Lift Coefficient vs Angle of Attack'),
            'cd_alpha': ('Alpha', 'Cd', 'Angle of Attack, α [deg]', 'Drag Coefficient, Cd [-]', 'Drag Coefficient vs Angle of Attack'),
            'cl_cd': ('Cd', 'Cl', 'Drag Coefficient, Cd [-]', 'Lift Coefficient, Cl [-]', 'Lift vs Drag Coefficient (Polar Diagram)'),
            'cm_alpha': ('Alpha', 'Cm', 'Angle of Attack, α [deg]', 'Moment Coefficient, Cm [-]', 'Moment Coefficient vs Angle of Attack')
        }
        
        if plot_type in plot_options:
            x_key, y_key, x_label, y_label, title = plot_options[plot_type]
            plt.plot(data[x_key], data[y_key], label=f'AF{af_index:02d}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
    
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plot_type}_plot.png')  # Save the plot
    plt.show()

# Main execution block
if __name__ == "__main__":
    polar_data = import_polar_data()
    
    if polar_data:
        for plot_type in ['cl_alpha', 'cd_alpha', 'cl_cd', 'cm_alpha']:
            plot_polar_data(polar_data, plot_type=plot_type)
