# Third-party imports
import numpy as np

class Integrate:
    """Integrates reach-level FLPE algorithm data.

    Attributes
    ----------
    alg_dict: dict
        dictionary of algorithm data stored by algorithm name as numpy arrays
    basin_dict: dict
        dict of reach_ids and SoS file needed to process entire basin of data
    integ_dict: dict
        dict of integrator estimate data
    moi_params: ??
        ??
    stage1_estimates: ??
        ??
    sos_dict: dict
        dictionary of SoS data

    Methods
    -------
    integrate()
        integrate and store reach-level data
    """

    def __init__(self, alg_dict, basin_dict, sos_dict):
        """
        Parameters
        ----------
        alg_dict: dict
            dictionary of algorithm data stored by algorithm name as numpy arrays
        basin_dict: dict
            dict of reach_ids and SoS file needed to process entire basin of data
        sos_dict: dict
            dictionary of SoS data
        """

        self.alg_dict = alg_dict
        self.basin_dict = basin_dict
        self.integ_dict = {
            "q_mean": np.array([]),
            "flpe": {
                "geobam" : np.array([]),
                "hivdi" : np.array([]),
                "metroman" : np.array([]),
                "momma" : np.array([]),
                "sad" : np.array([])
            }
        }
        self.moi_params = None
        self.stage1_estimates = None
        self.sos_dict = sos_dict

    def integrate(self):
        """Integrate reach-level FLPE data.
        
        TODO: Implement

        Parameters
        ----------
        """

        raise NotImplementedError