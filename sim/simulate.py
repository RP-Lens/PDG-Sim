from numpy import array

from sim.dynamics import rk4_step

def simulate(b, gd, env, X0, t_max, dt):
    # Time in seconds
    t = 0

    # Pre-allocate lists to store simulation data
    time_history, X_history, throttle_history = [], [], []

    # Initialize the state vector
    X = X0.copy()

    ## Main simulation loop; while the time is less than the maximum time and the altitude is greater than zero
    while t < t_max and X[2] > 0:

        # Store simulation data
        time_history.append(t)
        X_history.append(X.copy())

        # Guidance logic to determine throttle setting and direction
        u = gd.command(t, X, b)
        
        # Store throttle setting for this step
        throttle_history.append(u[0]) 
        
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
    time_history.append(t)
    X_history.append(X.copy())
    throttle_history.append(0.0)
    X_history = array(X_history)

    return time_history, X_history, throttle_history

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

