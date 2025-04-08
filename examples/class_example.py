import sys
import os

# Add the 'inputs' folder to sys.path so that it can find bem_model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inputs')))

from bem_model import BEMTurbineModel

import matplotlib.pyplot as plt

def main():
    # Initialize the BEM model with data files
    bem_model = BEMTurbineModel(
        blade_file='inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat',
        operational_file='inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt',
        polar_dir='inputs/IEA-15-240-RWT/Airfoils/'
    )
    
    # Compute performance at a specific wind speed
    V0 = 10  # m/s
    performance = bem_model.compute_rotor_performance(V0)
    print(f"At {V0} m/s wind speed:")
    print(f"Power: {performance['power']/1e6:.2f} MW")
    print(f"Thrust: {performance['thrust']/1e3:.2f} kN")
    print(f"Power coefficient: {performance['CP']:.3f}")
    
    # Compute and plot power curve
    power_curve = bem_model.compute_power_curve()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(power_curve['V0'], power_curve['power']/1e6)
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power [MW]')
    plt.title('Power Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(power_curve['V0'], power_curve['thrust']/1e3)
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Thrust [kN]')
    plt.title('Thrust Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()