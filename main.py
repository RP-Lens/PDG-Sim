from dataclasses import dataclass
import numpy as np
from numpy import column_stack, pi, linalg, array

# Function to calculate atmospheric pressure and density at a given altitude
def atm(h):
    if h > 11000:
        raise ValueError("Altitude out of range for this model (0-11000 m).")        
    else:
      h = max(h, 0.0)  # Count negative altitudes as sea level
      p_atm = 101325 * (1 - 2.25577e-5 * h) ** 5.25588
      rho_atm = 1.225 * (1 - 2.25577e-5 * h) ** 4.25588
    return p_atm, rho_atm

# Class to keep track of booster specs
@dataclass
class Booster:
    m_dry: float      # Dry mass, kg   
    m_prop: float     # Propellant mass, kg
    mdot_max: float   # Maximum propellant mass flow rate, kg/s
    T_vac: float      # Vacuum thrust of engine, N
    A_e: float        # Nozzle exit area, m^2
    Cd: float         # Booster drag coefficient
    A_ref: float      # Reference area of booster, m^2 
    
    # Method to calculate thrust at a given throttle value and altitude
    def thrust(self, throttle, p_atm) -> float:   
      if throttle == 0.0:
          return 0.0  
      return throttle * self.T_vac - (p_atm * self.A_e)
    
    # Method to calculate mass flow rate at a given throttle value
    def mdot(self, throttle) -> float:
      return throttle * self.mdot_max

# Class to keep track of guidance parameters
@dataclass
class Guidance:
    t_ign: float
    throttle_cmd: float = 1.0
    phase: str = "coast"          # "coast" -> "burn" -> "off"
    shutdown_reason: str = ""

    def command(self, t, X, b):
      v, m = X[3:6], X[6]

      speed = np.linalg.norm(v)
      u_hat = -v / speed if speed > 1e-3 else np.array([0.0, 0.0, 1.0])

      # Phase changes: one way only, so the engine can never relight
      if self.phase == "coast" and t >= self.t_ign:
        # Coast finished; start burn
        self.phase = "burn"
      if self.phase == "burn":
        # If burning, check for propellant depletion
        if m <= b.m_dry + 1e-6:
            self.phase = "off"
            self.shutdown_reason = "propellant depleted"
        # Burning but z velocity positive
        elif v[2] >= 0:
            self.phase = "off"
            self.shutdown_reason = "stopped above ground"
      # Only burn if in the burn phase
      throttle = self.throttle_cmd if self.phase == "burn" else 0.0
      return throttle, u_hat

#TODO: Dynamics calculation
def dynamics(t, X, u, b):
    # Unpack state vector
    x, y, z, vx, vy, vz, m = X
    throttle, u_hat = u
    
    # Calculate atmospheric pressure and density at current altitude
    p_atm, rho_atm = atm(z)
    
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
    g = 9.80665  # m/s^2
    a_gravity = array([0.0, 0.0, -g])
    
    # Total acceleration
    a_total = a_thrust + a_drag + a_gravity
    
    # Return the derivatives of the state vector
    return array([vx, vy, vz, a_total[0], a_total[1], a_total[2], -mdot])

# Perform a single Runge-Kutta 4th order step
def rk4_step(f, t, X, u, b, dt):
    k1 = f(t, X, u, b)
    k2 = f(t + dt / 2, X + dt * k1 / 2, u, b)
    k3 = f(t + dt / 2, X + dt * k2 / 2, u, b)
    k4 = f(t + dt, X + dt * k3, u, b)
    return X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
  
# Interpolate to find the exact time of touchdown (z=0)
def interpolate_touchdown(t_hist, X_hist):
    z = X_hist[:, 2]
    if z[-1] > 0: # Not enough data to interpolate
        return None  
    if len(z) < 2:  # Not enough data to interpolate; started at or aboce ground
        return t_hist[-1], X_hist[-1]
      
    z0, z1 = z[-2], z[-1]
    s = z0 / (z0 - z1)  # Fraction of the last time step where z crossed zero
    t_td = t_hist[-2] + s * (t_hist[-1] - t_hist[-2])
    X_td = X_hist[-2] + s * (X_hist[-1] - X_hist[-2])
    X_td[2] = 0.0                 # remove rounding residue
    return t_td, X_td

if __name__ == "__main__":
    
    # Intialize booster & guidance classes
    b = Booster(m_dry=25600, m_prop=8000, mdot_max=306, T_vac=914e3, A_e=0.67, Cd=0.5, A_ref=(pi* 3.7**2)/4)
    gd = Guidance(t_ign = 15.0)
    
    # X is the state vector
    X  = array([0.0, 0.0, 3000.0,           # Initial position in m
                0.0, 0.0, 0.0,              # Initial velocity in m/s
                b.m_dry + b.m_prop])  # Initial mass in kg

    # Time controls in seconds
    t = 0
    dt = 0.01
    t_max = 500
    
    # Pre-allocate lists to store simulation data
    time_history, X_history, throttle_history = [], [], []
    
    # Simulation loop; while the time is less than the maximum time and the altitude is greater than zero
    while t < t_max and X[2] > 0:
      # Store simulation data
      time_history.append(t)
      X_history.append(X.copy())

      # Guidance logic to determine throttle setting and direction
      u = gd.command(t, X, b)
      
      throttle_history.append(u[0])  # Store throttle setting for this step
      
      # Perform RK4 integration step with throttle settings
      X = rk4_step(dynamics, t, X, u, b, dt)
      
      # Increment time
      t += dt
        
    # Save final state to history
    time_history.append(t)
    X_history.append(X.copy())
    throttle_history.append(0.0)
    
    X_history = array(X_history)

    # quick export by Claude xox
    time_history = array(time_history)
    X_history = array(X_history)
    throttle_history = array(throttle_history)

    speed = linalg.norm(X_history[:, 3:6], axis=1)
    data = column_stack([time_history, X_history, speed, throttle_history])
    header = "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,m_kg,speed_mps,throttle"
    np.savetxt("output/sim_output.csv", data, delimiter=",",
              header=header, comments="", fmt="%.6f")

    td = interpolate_touchdown(time_history, X_history)
    if td is None:
        print("No touchdown before t_max")
    else:
        t_td, X_td = td
        print(f"Touchdown at t = {t_td:.3f} s")
        print(f"  velocity = {X_td[3:6]} m/s, speed = {linalg.norm(X_td[3:6]):.2f} m/s")
        print(f"  propellant left = {X_td[6] - b.m_dry:.1f} kg")
        print("Engine phase at end: ", gd.phase + " - " + gd.shutdown_reason)