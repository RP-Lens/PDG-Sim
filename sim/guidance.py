from dataclasses import dataclass
import numpy as np

IGN_TOL = 1e-9  # Ignition timing tolerance

# Class to keep track of guidance parameters
@dataclass
class Guidance:
    t_ign: float
    throttle_cmd: float = 1.0
    phase: str = "coast"          # "coast" -> "burn" -> "off"
    shutdown_reason: str = ""

    # Use to determine exact throttle timestep to ignite throttle
    def max_step(self, t, X, b, dt):
        if self.phase == "coast" and t < self.t_ign - IGN_TOL:
            # Shorten timestep to ignite throttle if timestep bigger than time to ignition
            return min(dt, self.t_ign - t)
        return dt

    def command(self, t, X, b):
        v, m = X[3:6], X[6]

        speed = np.linalg.norm(v)
        u_hat = -v / speed if speed > 1e-3 else np.array([0.0, 0.0, 1.0])

        # Phase changes: one way only, so the engine can never relight
        if self.phase == "coast" and t >= self.t_ign - IGN_TOL:
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