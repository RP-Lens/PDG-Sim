# The convex trajectory generator to be used by guidance
from dataclasses import dataclass
import cvxpy as cp
import numpy as np

# A dataclass to hold the convex trajectory plan
@dataclass
class Plan:
    tf: float
    N: int
    dt: float
    u: np.ndarray
    fuel: float
    r: np.ndarray
    v: np.ndarray
    m: np.ndarray

# Convex solve for a fixed burn duration tf
def solve_fixed_tf(X, b, env, tf, N):
    # Unpack the state vector
    r0, v0, m0 = X[0:3], X[3:6], X[6]

    # -- Constants --
    dt = tf/N
    rho1 = b.eta_min * b.T_vac
    rho2 = b.eta_max * b.T_vac
    alpha = b.mdot_max / b.T_vac
    g = np.array([0, 0, -env.g])
    
    # Reference log-mass (full thrust from ignition) and Taylor coefficients, one per interval
    t = np.arange(N) * dt
    z0 = np.log(m0 - alpha * rho2 * t)
    mu1 = rho1 * np.exp(-z0)
    mu2 = rho2 * np.exp(-z0)
    
    # -- Variables --
    r = cp.Variable((3, N+1))
    v = cp.Variable((3, N+1))
    z = cp.Variable(N+1)
    u = cp.Variable((3, N))
    sigma = cp.Variable(N)
    
    # -- Constraints --
    cons = []
    
    # Boundary conditions
    cons += [r[:, 0] == r0, v[:, 0] == v0, z[0] == np.log(m0)]      # Initial state
    cons += [r[:, N] == 0, v[:, N] == 0, z[N] >= np.log(b.m_dry)]   # Final state (touchdown)
    
    # Dynamics constraints
    cons += [v[:, 1:] == v[:, :N] + dt * (u + g[:, None])]                                      # Velocity
    cons += [r[:, 1:] == r[:, :N] + dt * v[:, :N] + 0.5 * dt**2 * (u + g[:, None])]             # Position
    cons += [z[1:] == z[:N] - dt * alpha * sigma]                                               # Mass
    
    # Slack variable
    cons += [cp.norm(u, axis=0) <= sigma]
    
    # Throttle Constraints
    delta = z[:N] - z0
    cons += [sigma >= cp.multiply(mu1, 1 - delta + 0.5 * cp.square(delta))]   # min thrust
    cons += [sigma <= cp.multiply(mu2, 1 - delta)]                             # max thrust

    # Ground
    cons += [r[2, :] >= 0]

    # Mass bounds
    cons += [z[:N] >= z0]                                  # no lighter than full thrust throughout
    cons += [z[:N] <= np.log(m0 - alpha * rho1 * t)]       # no heavier than min thrust throughout
    
    # -- Solve --
    prob = cp.Problem(cp.Maximize(z[N]), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != "optimal":
        print("Solver status:", prob.status)
        return None

    # Temporary return until the Plan dataclass exists
    m = np.exp(z.value)
    return {"r": r.value, "v": v.value, "m": m, "u": u.value,
            "sigma": sigma.value, "fuel": m0 - m[-1], "dt": dt}

# def plan(X, b, env, tf_candidates, N):
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sim.setup import parse

    b, gd, env, X0, _ = parse("./config/test_cases/mars_lander_convex.toml")
    tf, N = 75.0, 100
    res = solve_fixed_tf(X0, b, env, tf, N)

    if res is None:
        print("No solution")
    else:
        u_norm = np.linalg.norm(res["u"], axis=0)
        throttle = u_norm * res["m"][:N] / b.T_vac
        print(f"Fuel used:      {res['fuel']:.1f} kg")
        print(f"Max slack gap:  {np.max(res['sigma'] - u_norm):.2e}")
        print(f"Throttle range: {throttle.min():.2f} to {throttle.max():.2f}")

        t_int = np.arange(N) * res["dt"]        # interval start times
        t_node = np.arange(N + 1) * res["dt"]   # node times
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
        ax1.step(t_int, throttle, where="post")
        ax1.axhline(b.eta_min, ls="--", c="k", lw=0.8)
        ax1.axhline(b.eta_max, ls="--", c="k", lw=0.8)
        ax1.set_ylabel(r"Throttle $\eta$ [-]")
        ax2.plot(t_node, res["r"][2])
        ax2.set_ylabel(r"Height $z$ [m]")
        ax2.set_xlabel("Time since ignition [s]")
        fig.tight_layout()
        plt.show()