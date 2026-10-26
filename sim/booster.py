# Class to keep track of booster specs
from dataclasses import dataclass
import numpy as np
from numpy import column_stack, pi, linalg, array

@dataclass
class Booster:
    m_dry: float      # Dry mass, kg   
    m_prop: float     # Propellant mass, kg
    mdot_max: float   # Maximum propellant mass flow rate, kg/s
    T_vac: float      # Vacuum thrust of engine, N
    A_e: float        # Nozzle exit area, m^2
    Cd: float         # Booster drag coefficient
    A_ref: float      # Reference area of booster, m^2 
    eta_min: float    # Minimum throttle setting (0.0 to 1.0)
    
    # Method to calculate thrust at a given throttle value and altitude
    def thrust(self, throttle, p_atm) -> float:   
        if throttle == 0.0:
            return 0.0  
        return throttle * self.T_vac - (p_atm * self.A_e)
    
    # Method to calculate mass flow rate at a given throttle value
    def mdot(self, throttle) -> float:
      return throttle * self.mdot_max