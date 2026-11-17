'''
Script to parse config file and build booser, guidance and environment objects.
Also returns initial state vector and timestep for simulation.
'''


import tomllib, pprint
from numpy import array

from sim.simulate import simData
from sim.guidance import Guidance
from sim.booster import Booster
from sim.dynamics import Environment

def parse(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    
    pprint.pprint(cfg)

    b = Booster(
        m_dry=cfg["booster"]["m_dry"],
        m_prop=cfg["booster"]["propellant"],
        mdot_max=cfg["booster"]["mdot_max"],
        T_vac=cfg["booster"]["T_vac"],
        Cd=cfg["booster"]["Cd"],
        A_ref=cfg["booster"]["A_ref"],
        A_e=cfg["booster"]["A_e"],
        eta_min=cfg["booster"]["eta_min"],
        eta_max=cfg["booster"]["eta_max"]
    )
    
    # TODO: error-catching for guidance type
    gd = Guidance(
        t_ign=cfg["guidance"]["t_ign"],
        guidance_type=cfg["guidance"]["type"],
        eta_min=b.eta_min,
        eta_max=b.eta_max
    )
    env = Environment(
        g=cfg["environment"]["g"],
        model=cfg["environment"]["model"]
    )
    
    # Take intial pos, vel & total mass and return state vector
    X0 = array([
        cfg["initial"]["position"][0], cfg["initial"]["position"][1], cfg["initial"]["position"][2],
        cfg["initial"]["velocity"][0], cfg["initial"]["velocity"][1], cfg["initial"]["velocity"][2],
        b.m_dry + b.m_prop
    ])
    
    dt = cfg["sim"]["dt"]

    return b, gd, env, X0, dt