import numpy as np 
#rho = 1.225  # Air density (kg/m^3)
#bl_radius = 240  # Rotor radius (meters)
#no_blades = 3  # Number of blades

###################################################
#   Intro
###################################################
def compute_local_solidity(span, blade_data, no_blades=3):
    #!!!!!!!!!!!! Interpolate blade_data to get c_r
    # local_solidity = (c_r * no_blades) / (2 * np.pi * span)
    # return local_solidity
    pass

def compute_tsr(rot_speed, v0, bl_radius = 240):
    tsr = (rot_speed * bl_radius) / v0
    return tsr

def get_opt_strategy(v0, opt_data):
    # interpolate to get theta_p, omega, power, thrust
    # return theta_p, omega, power, thrust
    pass

###################################################
#   Step 2 Equation
###################################################
def compute_flow_angle(a, a_prime, v0, rot_speed, span):
    flow_angle = np.arctan(((1 - a) * v0) / ((1 + a_prime) * rot_speed * span))
    return flow_angle

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
    #interpolate polar data to get cl and cd
    pass

###################################################
#   Step 5 Equation
###################################################
def compute_normal_tangential(cl, cd, flow_angle):
    cn = cl * np.cos(flow_angle) + cd * np.sin(flow_angle)
    ct = cl * np.sin(flow_angle) - cd * np.cos(flow_angle)
    return cn, ct

###################################################
#   Step 6 Equation
###################################################
def update_induction_factors(local_solidity, Cn, Ct, flow_angle):
    a_new = 1 / (4 * (np.sin(flow_angle) ** 2) / ((local_solidity * Cn) + 1))
    a_prime_new = 1 / (4 * (np.sin(flow_angle) * np.cos(flow_angle)) / ((local_solidity * Ct) - 1))
    return a_new, a_prime_new

###################################################
#   Step 8 Equation
###################################################
def compute_thrust_torque(a, a_prime, v0, rot_speed, span, dr, rho = 1.225):
    thrust = 4 * np.pi * span * rho * (v0 ** 2) * a * (1 - a) * dr
    torque = 4 * np.pi * (span ** 3) * rho * v0 * rot_speed * a_prime * (1 - a) * dr
    return thrust, torque

###################################################
#   Final Step Equation
###################################################
def compute_thrust_power(thrust, power, v0, rho = 1.225, bl_radius = 240):
    area = np.pi * (bl_radius ** 2)
    ct = thrust / (0.5 * rho * area * (v0 ** 2))
    cp = power / (0.5 * rho * area * (v0 ** 3))
    return ct, cp