import numpy as np
from numpy import array
from dataclasses import dataclass, field

from sim.dynamics import rk4_step

@dataclass
class simData:
    time_history: list[float] = field(default_factory=list)
    X_history: list[np.ndarray] = field(default_factory=list)
    throttle_history: list[float] = field(default_factory=list)
    shutdown_reason: str = ""
    
    # Interpolate to find exact time of touchdown (z=0)
    def interpolate_touchdown(self):
        z = self.X_history[:, 2]
        
        #Check if there is enough data to interpolate
        if z[-1] > 0:
            return None
        if len(z) < 2:
            return self.time_history[-1], self.X_history[-1]
        
        z0, z1 = z[-2], z[-1]
        
        # Fraction of time where z crossed 0
        s = z0 / (z0-z1)
        
        # Interpolate to find actual touchdown time and vehicle state
        t_td = self.time_history[-2] + s * (self.time_history[-1] - self.time_history[-2])
        X_td = self.X_history[-2] + s * (self.X_history[-1] - self.X_history[-2])
        
        # remove rounding residue
        X_td[2] = 0.0                
        return t_td, X_td

'''
    Function runs simulation loop. Tracks time - guidance and booster can shorten timestep if needed.
    
    Simulation state saved to simData object
''' 

def simulate(b, gd, sd, env, X0, t_max, dt):
    # Time in seconds
    t = 0

    # Initialize the state vector
    X = X0.copy()

    ## Main simulation loop; while the time is less than the maximum time and the altitude is greater than zero
    while t < t_max and X[2] > 0:

        # Store simulation data
        sd.time_history.append(t)
        sd.X_history.append(X.copy())

        # Guidance logic to determine throttle setting and direction
        u = gd.command(t, X, b)
        
        # Store throttle setting for this step
        sd.throttle_history.append(u[0]) 
        
        ## Time control
        # Don't step past ignition
        step = gd.max_step(t, X, b, dt)
        
        # Don't step past propellant depletion
        if u[0] > 0:
            # t_left = (m - m_dry) / mdot(throttle)          
            step = min(step, (X[6] - b.m_dry) / b.mdot(u[0])) 

        # Perform RK4 integration step with throttle settings & atmosphere model
        X = rk4_step(t, X, u, b, step, env)
        
        # Increment time
        t += step
        
    # Save final state to history
    sd.time_history.append(t)
    sd.X_history.append(X.copy())
    sd.throttle_history.append(0.0)
    sd.X_history = array(sd.X_history)



