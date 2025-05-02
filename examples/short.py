"""Example script demonstrating the direct use of the package's main functions.
It allows obtaining results in a straightforward way without intermediate steps."""
from windbem import BEMTurbineModel
from windbem import compute_rotor_performance
from windbem import plot_results
from windbem import print_results

bem_model = BEMTurbineModel(
    blade_file ='inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat',
    operational_file ='inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt',
    polar_dir ='inputs/IEA-15-240-RWT/Airfoils/'
   )

V0 = 10  # m/s

performance = compute_rotor_performance(bem_model, V0)
bem_model.plot_af_shapes()
plot_results(bem_model)
print_results(performance)
