from dataclasses import dataclass
import numpy as np
from numpy import column_stack, pi, linalg, array
import pprint
import matplotlib.pyplot as plt

from sim.simulate import simulate, simData
from sim.guidance import Guidance
from sim.booster import Booster
from sim.dynamics import Environment
from sim.setup import parse

if __name__ == "__main__":
    
    # Parse TOML file & initialise simulation parameters
    path="./config/test_cases/freefall_vac_3k.toml"
    b, gd, env, X0, dt = parse(path)
    sd = simData()

    # Perform simulation    
    simulate(b, gd, sd, env=env, X0=X0, t_max=500, dt=dt)

    # Find exact touchdown time & guidance state at touchdown
    td = sd.interpolate_touchdown()
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
    time_history = array(sd.time_history)
    X_history = array(sd.X_history)
    throttle_history = array(sd.throttle_history)

    speed = linalg.norm(X_history[:, 3:6], axis=1)
    data = column_stack([time_history, X_history, speed, throttle_history])
    header = "t_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,m_kg,speed_mps,throttle"
    np.savetxt("output/freefall_vac_3k.csv", data, delimiter=",",
        header=header, comments="", fmt="%.6f")
    
    d = np.genfromtxt("output/freefall_vac_3k.csv", delimiter=",", names=True)
    t = d["t_s"]

    fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))


    # Panel 1: height and vertical velocity
    c1, c2 = "tab:blue", "tab:red"
    ax1.plot(t, d["z_m"], color=c1)
    ax1.set_ylabel(r"Height $z$ [m]", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.plot(t, d["vz_mps"], color=c2)
    ax2.axhline(0, color=c2, lw=0.6, ls=":")
    ax2.set_ylabel(r"$v_z$ [m/s]", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title("Height and vertical velocity")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: mass and throttle
    c3, c4 = "tab:green", "tab:orange"
    ax3.plot(t, d["m_kg"], color=c3)
    ax3.axhline(b.m_dry, color=c3, lw=0.8, ls="--")  # dry mass reference
    ax3.set_ylabel(r"Booster mass $m$ [kg]", color=c3)
    ax3.tick_params(axis="y", labelcolor=c3)
    ax4 = ax3.twinx()
    ax4.plot(t, d["throttle"], color=c4)
    ax4.set_ylabel(r"Throttle $ eta$ [-]", color=c4)
    ax4.tick_params(axis="y", labelcolor=c4)
    ax3.set_title("Mass and throttle setting")
    ax3.set_xlabel(r"Time $t$ [s]")
    ax3.grid(True, alpha=0.3)

    # Mark touchdown
    if td is not None:
        for ax in (ax1, ax3):
            ax.axvline(t_td, color="k", lw=0.8, ls="--")

    fig.tight_layout()
    fig.savefig("output/freefall_vac_3k.png", dpi=200)
    plt.show()
