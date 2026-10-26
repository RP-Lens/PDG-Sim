from numpy import linalg, array
from dataclasses import dataclass

@dataclass
class Environment:
    g: float  # m/s^2
    model: str  # Atmosphere model: 'ISA' or 'vacuum'


# Function to calculate atmospheric pressure and density at a given altitude
def atm(h, env):
    # Can use ISA model or vacuum
    if env.model == 'ISA':
        if h > 11000:
            raise ValueError("Altitude out of range for this model (0-11000 m).")        
        else:
            h = max(h, 0.0)  # Count negative altitudes as sea level
            p_atm = 101325 * (1 - 2.25577e-5 * h) ** 5.25588
            rho_atm = 1.225 * (1 - 2.25577e-5 * h) ** 4.25588
    else:        
        p_atm, rho_atm = 0.0, 0.0
    return p_atm, rho_atm

# Dynamics calculation
def dynamics(t, X, u, b, env):
    # Unpack state vector
    x, y, z, vx, vy, vz, m = X
    throttle, u_hat = u
    
    # Calculate atmospheric pressure and density at current altitude
    p_atm, rho_atm = atm(z, env)
    
    # Calculate thrust and mass flow rate based on throttle setting
    T = b.thrust(throttle, p_atm)
    mdot = b.mdot(throttle)
    
    # Calculate drag force
    v = array([vx, vy, vz])
    speed = linalg.norm(v)
    F_drag = -0.5 * rho_atm * b.Cd * b.A_ref * speed * v 
    
    # Calculate acceleration due to thrust and drag
    a_thrust = (T / m) * u_hat
    a_drag = F_drag / m
    
    # Gravitational acceleration (assuming constant g for simplicity)
    a_gravity = array([0.0, 0.0, -env.g])
    
    # Total acceleration
    a_total = a_thrust + a_drag + a_gravity
    
    # Return the derivatives of the state vector
    return array([vx, vy, vz, a_total[0], a_total[1], a_total[2], -mdot])

# Perform a single Runge-Kutta 4th order step
def rk4_step(t, X, u, b, dt, env):
    k1 = dynamics(t, X, u, b, env)
    k2 = dynamics(t + dt / 2, X + dt * k1 / 2, u, b, env)
    k3 = dynamics(t + dt / 2, X + dt * k2 / 2, u, b, env)
    k4 = dynamics(t + dt, X + dt * k3, u, b, env)
    return X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)