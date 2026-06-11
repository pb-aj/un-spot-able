"""
File set up configuration class used in fitclass.py
Code sourced from: https://github.com/rychallener/theresa
Challener, R. C., & Rauscher, E. 2022, AJ, 163, 117, doi: 10.3847/1538-3881/ac4885
"""

class Configuration:
    """
    A class to hold parameters from a configuration file.
    """
    def __init__(self):
        self.planet = Planet()
        self.star   = Star()
        self.sim   = Sim()

class Planet:
    """
    A class to hold planet parameters.
    """
    pass

class Star:
    """
    A class to hold star parameters.
    """
    pass

class Sim:
    """
    A class to hold simulation configuration options.
    """
    pass
