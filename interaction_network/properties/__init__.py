import pandas as pd
from add_halos import EstablishNetwork

class BaseProperties(EstablishNetwork):

    names = []

    def __init__(self, halo, config_filename=None, verbose=False):
        super().__init__(config_filename=config_filename, verbose=verbose)
        self.network = self.read_network(halo)

    def read_network(self, halo):
        filename = self.sim_config["OUT_FILE"]
        df = pd.read_hdf(filename, key="r" + str(halo))
        return df

    def calculate(self, halo):
        """Generate interaction parameters for given halo

        Args:
            halo_num (int): halo number at desired step
        """
        return NotImplementedError

    def save_network(self, halo, data, names):
        """Save to disk

        Args:
            halo_num (int): Halo number
            df (pandas.DataFrame): DataFrame containing interaction network
        """
        df = self.network
        df[names] = data
        df.to_hdf(self.sim_config["OUT_FILE"], key="r" + str(halo))