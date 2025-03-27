# wk 8 importing and plotting homework
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


# Irene - IEA-15-240-RWT_AeroDyn15_Polar_00.dat
# Hubert - Airfoils/IEA-15-240-RWT_AF00_Coords.txt
