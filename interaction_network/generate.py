import argparse
import glob
import json
import pathlib
from json.decoder import JSONDecodeError

import numpy as np
import pandas as pd
import tangos as db
import tqdm

CONFIG = pathlib.Path() / "config.json"


def read_json(file_path):
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except (FileNotFoundError, JSONDecodeError) as err:
            print(err, "Invalid JSON config")


def get_halos(filename):
    """Generate list of halos from input file

    Args:
        filename (str): list of halos, one per line

    Returns:
        numpy.ndarray: list of halo number integers
    """
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    return [int(h) for h in lines]


class DwarfInteractionNetwork:
    def __init__(self, config_filename=None, verbose=False):
        if config_filename is None:
            config = read_json(CONFIG)
        else:
            config = read_json(config_filename)

        self.sim_config = config["SIMULATION"]
        self.dim_config = config["DIN"]
        self.verbose = bool(verbose)

        self.ahf_filenames = self.get_ahf_filenames()
        self.min_ndm = float(self.dim_config["min_ndm"])
        self.max_dist = float(self.dim_config["max_neighbor_physical_dist"])
        self.L = float(self.dim_config["boxsize_comoving"])
        self.h = float(self.dim_config["h"])

        self.ahf_halos = self.read_ahf_halos()
        self.save_ahf_halos()

        self.names = [
            "neighbor_halo_num",
            "mvir",
            "theta",
            "dist",
            "halo_num",
            "timestep",
            "z",
        ]

    def get_ahf_filenames(self):
        """Gather filenames of *AHF_halos files from simulation directory

        Returns:
            list: *AHF_halos filenames
        """
        ahf_filenames = glob.glob(str(self.sim_config["SIM_DIR"]) + "*.AHF_halos")
        ahf_filenames.sort(reverse=True)
        if self.verbose:
            return tqdm.tqdm(ahf_filenames)
        else:
            return ahf_filenames

    def get_timestep_number(self, halo):
        """Split snapshot filename to get timestep

        Args:
            halo (tangos.halo): Halo object from Tangos

        Returns:
            int: Timestep number
        """
        return int(halo.timestep.extension.split(".")[-1])

    def periodic_wrap(self, halo_coords, neighbor_coords, boxsize):
        """Wrap coordinates across [-L/2, L/2]

        Args:
            halo_coords (numpy.ndarray(N,1)): Coordinates of halo of interest
            neighbor_coords (numpy.ndarray(N,M)): Coordinates of all halos
            boxsize (float): Size of box such that it spans [-L/2, L/2]

        Returns:
            Relative coordinates: Relative, periodically wrapped coordinates
        """
        relpos = neighbor_coords - halo_coords
        wrap_ix = np.where(np.abs(relpos) > boxsize / 2.0)
        relpos[wrap_ix] = (
            -1.0
            * (relpos[wrap_ix] / np.abs(relpos[wrap_ix]))
            * np.abs(boxsize - np.abs(relpos[wrap_ix]))
        )
        return relpos

    def read_ahf_halos(self):
        """Read halo properties from all present *AHF_halos files from Amiga Halo Finder

        Returns:
            pandas.DataFrame: DataFrame containing all AHF_halos data
        """
        try:
            return pd.read_hdf(self.sim_config["OUT_AHF"], key="data")
        except OSError:
            return pd.concat(
                [self.get_ahf_props(filename) for filename in self.ahf_filenames]
            )

    def save_ahf_halos(self):
        self.ahf_halos.to_hdf(self.sim_config["OUT_AHF"], key="data")

    def get_ahf_props(self, filename):
        """Gather properties from *.AHF_halos file into Pandas DataFrame

        Args:
            filename (str): AHF_halos csv full path
            ndm (int): Minimum number of dark matter particles to consider

        Returns:
            pandas.DataFrame: DataFrame generated from 'filename'
        """
        df = (
            pd.read_csv(
                filename,
                delim_whitespace=True,
                header=0,
                usecols=["Mvir(4)", "npart(5)", "Xc(6)", "Yc(7)", "Zc(8)"],
            )
            .where(lambda x: x["npart(5)"] >= self.min_ndm)
            .dropna()
            .sort_values(by="npart(5)", ascending=False)
            .rename(columns={"Mvir(4)": "mvir", "npart(5)": "ndm"})
            .assign(
                timestep=lambda x: int(filename.split(".")[-4]),
                x=lambda x: x["Xc(6)"] / self.h,
                y=lambda x: x["Yc(7)"] / self.h,
                z=lambda x: x["Zc(8)"] / self.h,
            )
            .drop(columns=["Xc(6)", "Yc(7)", "Zc(8)"])
            .reset_index()
        )
        return df

    def query(self, halo_num_original):
        """Generate interaction parameters for given halo

        Args:
            halo_num (int): halo number at desired step
            L (float, optional): boxsize in cMpc for periodic boundary conditions. Defaults to 25e3.
        """
        snapshot_name = (
            str(self.sim_config["TANGOS_DB"])
            + "/%"
            + str(self.sim_config["TANGOS_ORIGINAL_TIMESTEP"])
        )
        snapshot_original = db.get_timestep(snapshot_name)
        halo_original = snapshot_original[halo_num_original]

        halo = halo_original
        neighbor_props = []
        while halo is not None:
            step_num = self.get_timestep_number(halo)
            z = halo.timestep.redshift
            neighbors = self.get_nearby_neighbors(halo)
            if len(neighbors) > 0:
                row = self.calc_neighbor_props(neighbors)
                row += [halo_original.halo_number, step_num, z]
                neighbor_props.append(row)
            halo = halo.previous
        return np.vstack(neighbor_props)

    def get_nearby_neighbors(self, halo):
        """Calculate nearest neighbors within some max distance, and their distances

        Args:
            halo (tangos.halo): Tangos halo object

        Returns:
            pandas.DataFrame: Dataframe containing nearest neighbors at a given timestep
        """
        try:
            halo_coords = halo["shrink_center"]
        except KeyError:
            return pd.DataFrame()
        a = 1 / (1 + halo.timestep.redshift)
        max_dist_comoving = self.max_dist / a
        step_num = self.get_timestep_number(halo)

        neighbors = self.ahf_halos.query("timestep == @step_num").reset_index(drop=True)
        neighbor_coords = neighbors[["x", "y", "z"]].values
        neighbor_coords_relative = self.periodic_wrap(
            halo_coords, neighbor_coords, self.L
        )
        nearest_neighbor_dist = np.linalg.norm(neighbor_coords_relative, axis=1)
        neighbors["r"] = nearest_neighbor_dist
        nearest_ix = neighbors["r"] <= max_dist_comoving

        return neighbors.loc[nearest_ix]

    def calc_neighbor_props(self, neighbors):
        """Calculate properties of most significant nearest neighbor, and number of nearest neighbors

        Args:
            neighbors (pandas.DataFrame): DataFrame containing neighbors from AHF
            neighbor_dist (numpy.ndarray(N,1)): Distances to nearest neighbors

        Returns:
            list: list containing nearest neighbor properties
        """
        halo_num = self.calc_halo_number(neighbors)
        mvir = self.calc_halo_mvir(neighbors)
        theta = self.calc_tidal_index(neighbors)
        dist = self.calc_distance_neighbor(neighbors)

        most_significant_neighbor = np.argmax(theta)
        return [
            halo_num[most_significant_neighbor],
            mvir[most_significant_neighbor],
            theta[most_significant_neighbor],
            dist[most_significant_neighbor],
        ]

    def calc_halo_mvir(self, neighbors):
        """Virial mass of neighbor at current timestep"""
        return neighbors["mvir"].values

    def calc_halo_number(self, neighbors):
        """Halo number of neighbor at current timestep"""
        return neighbors["index"].values + 1

    def calc_tidal_index(self, neighbors):
        """Calculate tidal index according to Karachentsev+(1999)"""
        mvir = neighbors["mvir"].values
        dist = neighbors["r"].values
        tidal_indices = np.log10(mvir / (1e-3 * dist) ** 3)
        return tidal_indices - 11.75

    def calc_distance_neighbor(self, neighbors):
        return neighbors["r"].values

    def save_interaction_network(self, halo_num, data):
        """Save to disk

        Args:
            data (numpy.ndarray): Data array
        """
        df = pd.DataFrame(data, columns=self.names)
        df.to_hdf(self.sim_config["OUT_FILE"], key="r" + str(halo_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate halo interaction network for input Romulus25 halo at step 7779"
    )
    parser.add_argument("--halos", help="File containing halo numbers at step 7779")
    parser.add_argument("--verbose", help="Verbose", action="store_true")
    args = parser.parse_args()

    verbose = bool(args.verbose)
    halo_numbers = get_halos(args.halos)

    assert len(halo_numbers) > 0
    assert type(halo_numbers[0]) == int

    if verbose:
        halo_numbers = tqdm.tqdm(halo_numbers)

    din = DwarfInteractionNetwork(config_filename=CONFIG, verbose=verbose)

    for halo_number in halo_numbers:
        data = np.vstack(din.query(halo_number))
        din.save_interaction_network(halo_number, data)
