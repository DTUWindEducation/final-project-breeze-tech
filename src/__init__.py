import numpy as np 
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

#rho = 1.225  # Air density (kg/m^3)
#bl_radius = 120  # Rotor radius (meters)
#no_blades = 3  # Number of blades

###################################################
#   Intro
###################################################
def compute_local_solidity(span, blade_data, no_blades=3):
    c_r = np.interp(span, blade_data['span'], blade_data['chord'])
    
    local_solidity = (c_r * no_blades) / (2 * np.pi * span)
    return local_solidity

def compute_tsr(rot_speed, v0, bl_radius = 120):
    rot_speed_rad_s = rot_speed * 2 * np.pi / 60
    tsr = (rot_speed_rad_s * bl_radius) / v0
    return tsr

def get_opt_strategy(v0, opt_data):
    theta_p = np.interp(v0, opt_data['wind_speed'], opt_data['pitch'])
    omega = np.interp(v0, opt_data['wind_speed'], opt_data['rot_speed'])
    power = np.interp(v0, opt_data['wind_speed'], opt_data['power'])
    thrust = np.interp(v0, opt_data['wind_speed'], opt_data['thrust'])
    
    return theta_p, omega, power, thrust

###################################################
#   Step 2 Equation
###################################################
def compute_flow_angle(a, a_prime, v0, rot_speed, span):
    rot_speed_rad_s = rot_speed * 2 * np.pi / 60

    flow_angle_rad = np.arctan(((1 - a) * v0) / ((1 + a_prime) * rot_speed_rad_s * span))
    flow_angle_deg = np.rad2deg(flow_angle_rad)
    return flow_angle_deg

###################################################
#   Step 3 Equation
###################################################
def compute_angle_attack(flow_angle, pitch, bl_twist):
    angle_attack = flow_angle - (pitch + bl_twist)
    return angle_attack

###################################################
#   Step 4 Equation
###################################################
def compute_lift_drag(angle_attack, polar_data):
    polar_data_alphas, polar_data_cls, polar_data_cds = polar_data
    
    cl = np.interp(angle_attack, polar_data_alphas, polar_data_cls)
    cd = np.interp(angle_attack, polar_data_alphas, polar_data_cds)
    
    return cl, cd

###################################################
#   Step 5 Equation
###################################################
def compute_normal_tangential(cl, cd, flow_angle):
    flow_angle_rad = np.deg2rad(flow_angle)
    cn = cl * np.cos(flow_angle_rad) + cd * np.sin(flow_angle_rad)
    ct = cl * np.sin(flow_angle_rad) - cd * np.cos(flow_angle_rad)
    return cn, ct

###################################################
#   Step 6 Equation
###################################################
def update_induction_factors(local_solidity, cn, ct, flow_angle):
    flow_angle_rad = np.deg2rad(flow_angle)
    a_new = 1 / (4 * (np.sin(flow_angle_rad) ** 2) / ((local_solidity * cn) + 1))
    a_prime_new = 1 / (4 * (np.sin(flow_angle_rad) * np.cos(flow_angle_rad)) / ((local_solidity * ct) - 1))
    return a_new, a_prime_new

###################################################
#   Step 8 Equation
###################################################
def compute_dthrust_dtorque(a, a_prime, v0, rot_speed, span, dr, rho = 1.225):
    rot_speed_rad_s = rot_speed * 2 * np.pi / 60
    d_thrust = 4 * np.pi * span * rho * (v0 ** 2) * a * (1 - a) * dr
    d_torque = 4 * np.pi * (span ** 3) * rho * v0 * rot_speed_rad_s * a_prime * (1 - a) * dr
    return d_thrust, d_torque

###################################################
#   Final Step Equation
###################################################
def compute_thrust_power(thrust, power, v0, rho = 1.225, bl_radius = 120):
    area = np.pi * (bl_radius ** 2)
    thrust_coef = thrust / (0.5 * rho * area * (v0 ** 2))
    power_coef = power / (0.5 * rho * area * (v0 ** 3))
    return thrust_coef, power_coef


###################################################
#   BEM Solver
###################################################
def solve_bem(v0, theta_p, rot_speed, blade_data, airfoil_polars, 
              rho=1.225, bl_radius=120.0, no_blades=3, max_iter=100, tol=1e-6):
    spans = blade_data['span']
    
    a_values = []
    a_prime_values = []
    d_thrust_values = []
    d_torque_values = []
    
    for i in range(len(spans)):
        span = spans[i]
        
        # Calculate dr as the difference between the current and next span
        # For the last element keep the same dr or difference to the end of the  
        if (i < (len(spans)-1)):
            dr = spans[i+1] - spans[i]
        else:
            dr = dr
            #dr = bl_radius - spans[i]
        
        bl_twist = blade_data['twist'][i]
       
        # get the polar file corresponding to the current af_id
        # !!!!!!!!!!!!
        polar_data = airfoil_polars[i]
        
        local_solidity = compute_local_solidity(span, blade_data, no_blades)
        
        a = 0
        a_prime = 0
        
        for _ in range(max_iter):
            a_old = a
            a_prime_old = a_prime
            
            # Step 2: Compute flow angle
            flow_angle = compute_flow_angle(a, a_prime, v0, rot_speed, span)
            
            # Step 3: Compute angle of attack
            angle_attack = compute_angle_attack(flow_angle, theta_p, bl_twist)
            
            # Step 4: Compute lift and drag coefficients
            cl, cd = compute_lift_drag(angle_attack, polar_data)
            
            # Step 5: Compute normal and tangential coefficients
            cn, ct = compute_normal_tangential(cl, cd, flow_angle)
            
            # Step 6: Update induction factors
            a, a_prime = update_induction_factors(local_solidity, cn, ct, flow_angle)
            
            # Check for convergence
            if (abs(a - a_old) < tol) and (abs(a_prime - a_prime_old) < tol):
                break
        
        # Step 8: Compute thrust and torque
        d_thrust, d_torque = compute_dthrust_dtorque(a, a_prime, v0, rot_speed, span, dr, rho)
        
        a_values.append(a)
        a_prime_values.append(a_prime)
        d_thrust_values.append(d_thrust)
        d_torque_values.append(d_torque)
    
    total_thrust = sum(d_thrust_values)
    total_torque = sum(d_torque_values)
    total_power = total_torque * (rot_speed * 2 * np.pi / 60)  # Convert rpm to rad/s
    
    return {
        'span': spans,
        'a': a_values,
        'a_prime': a_prime_values,
        'thrust_elements': d_thrust_values,
        'torque_elements': d_torque_values,
        'total_thrust': total_thrust,
        'total_torque': total_torque,
        'total_power': total_power
    }
    
###################################################
#   Main Function
###################################################
def compute_power_thrust_curve(blade_data, opt_data, airfoil_polars, 
                              wind_speeds=None, bl_radius=120.0):
    if wind_speeds is None:
        wind_speeds = np.linspace(3, 25, 23)  # 3 to 25 m/s
    
    power_values = []
    thrust_values = []
    power_coef_values = []
    thrust_coef_values = []
    opt_power_values = []
    opt_thrust_values = []
    
    for v0 in wind_speeds:
        theta_p, rot_speed, opt_power, opt_thrust = get_opt_strategy(v0, opt_data)
        
        results = solve_bem(v0, theta_p, rot_speed, blade_data, airfoil_polars, bl_radius=bl_radius)
        
        power_values.append(results['total_power'] / 1000)  
        thrust_values.append(results['total_thrust'] / 1000)  
        
        thrust_coef, power_coef = compute_thrust_power(results['total_thrust'], results['total_power'], v0, bl_radius=bl_radius)
        power_coef_values.append(power_coef)
        thrust_coef_values.append(thrust_coef)
        
        opt_power_values.append(opt_power)
        opt_thrust_values.append(opt_thrust)
    
    return {
        'wind_speeds': wind_speeds,
        'power': np.array(power_values),
        'thrust': np.array(thrust_values),
        'cp': np.array(power_coef_values),
        'ct': np.array(thrust_coef_values),
        'opt_power': np.array(opt_power_values),
        'opt_thrust': np.array(opt_thrust_values)
    }