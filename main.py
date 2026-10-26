from dataclasses import dataclass
import numpy as np
from numpy import column_stack, pi, linalg, array

from sim.simulate import simulate, interpolate_touchdown
from sim.guidance import Guidance
from sim.booster import Booster
from sim.dynamics import Environment

if __name__ == "__main__":
    
    # TODO: TOML file for simulation parameters
    b = Booster(m_dry=25600, m_prop=8000, mdot_max=306, T_vac=914e3, A_e=0.67, Cd=0.5, A_ref=(pi* 3.7**2)/4, eta_min=0.6)
    gd = Guidance(t_ign = 20)
    env = Environment(g=9.80665, model='ISA')  # Use ISA atmosphere model
    
    # X is the state vector
    X0  = array([0.0, 0.0, 3000.0,           # Initial position in m
                0.0, 25.0, 0.0,             # Initial velocity in m/s
                b.m_dry + b.m_prop])        # Initial mass in kg

    # Perform simulation
    time_history, X_history, throttle_history = simulate(b, gd, env=env, X0=X0, t_max=500, dt=0.001)

    # Find exact touchdown time & guidance state at touchdown
    td = interpolate_touchdown(time_history, X_history)
    if td is None:
        print("No touchdown before t_max")
    else:
        t_td, X_td = td
        print(f"Touchdown at t = {t_td:.3f} s")
        print(f"  velocity = {X_td[3:6]} m/s, speed = {linalg.norm(X_td[3:6]):.2f} m/s")
        print(f"  propellant left = {X_td[6] - b.m_dry:.1f} kg")
        if gd.phase == "burn":
            print("Engine phase at end: stillll burnin'")
        else:
            print("Engine phase at end: ", gd.phase + " - " + gd.shutdown_reason)

    # quick export by Claude xox
    time_history = array(time_history)
    X_history = array(X_history)
    throttle_history = array(throttle_history)

    speed = linalg.norm(X_history[:, 3:6], axis=1)
    data = column_stack([time_history, X_history, speed, throttle_history])
    header = "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,m_kg,speed_mps,throttle"
    np.savetxt("output/sim_output.csv", data, delimiter=",",
        header=header, comments="", fmt="%.6f")
    