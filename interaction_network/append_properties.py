import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import tangos as db
import tqdm
import pynbody

import generate
from generate import DwarfInteractionNetwork


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


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


class AddSnapshotProperties(DwarfInteractionNetwork):
    def __init__(self, config_filename=None, verbose=False):

        super().__init__(config_filename=config_filename, verbose=verbose)

    def read_interaction_network(self, halo_num):
        filename = self.sim_config["OUT_FILE"]
        df = pd.read_hdf(filename, key="r" + str(halo_num))
        return df

    def query(self, halo_num):
        """Generate interaction parameters for given halo

        Args:
            halo_num (int): halo number at desired step
        """
        data = self.read_interaction_network(halo_num)
        data = self.calc_properties(data)
        return data

    def calc_properties(self, data):
        data["Mvir_max"] = self.calc_halo_mvir_max(data)
        data['Mstar_max'] = self.calc_halo_mstar_max(data)
        return data

    def calc_halo_mvir_max(self, data):
        """Iterate over each most significant neighbor, trace each of their progenitors back in time, and find max virial max from AHF

        Args:
            neighbors (pandas.DataFrame): *AHF_halos DataFrame
        """

        mvir_max_arr = []
        rows = data.iterrows()
        if self.verbose:
            rows = tqdm.tqdm(rows)

        for i, row in rows:
            snapshot_name = (
                str(self.sim_config["TANGOS_DB"]) + "/%00" + str(int(row["timestep"]))
            )
            neighbor_halo = db.get_halo(
                snapshot_name + "/" + str(int(row["neighbor_halo_num"]))
            )
            if neighbor_halo is not None:
                progs = db.relation_finding.MultiHopMajorProgenitorsStrategy(
                    neighbor_halo
                ).all()
                mvir = []
                for prog in progs:
                    prog_halo_num = prog.halo_number
                    prog_timestep = self.get_timestep_number(prog)
                    prog_ix = (self.ahf_halos["index"] == prog_halo_num) & (
                        self.ahf_halos["timestep"] == prog_timestep
                    )
                    mvir.append(self.ahf_halos.loc[prog_ix, "mvir"])

                try:
                    mvir_max = np.hstack(mvir).max()
                except ValueError:
                    mvir_max = 0.0
            else:
                mvir_max = 0.0

            mvir_max_arr.append(mvir_max)
        return np.asarray(mvir_max_arr)

    def calc_halo_mstar_max(self, data):
        """Iterate over each most significant neighbor, trace each of their progenitors back in time, and find max stellar max from AHF

        Args:
            neighbors (pandas.DataFrame): *AHF_halos DataFrame
        """

        mstar_max_arr = []
        rows = data.iterrows()
        if self.verbose:
            rows = tqdm.tqdm(rows)

        for i, row in rows:
            snapshot_name = (
                str(self.sim_config["TANGOS_DB"]) + "/%00" + str(int(row["timestep"]))
            )
            neighbor_halo = db.get_halo(
                snapshot_name + "/" + str(int(row["neighbor_halo_num"]))
            )

            if neighbor_halo is not None:
                progs = db.relation_finding.MultiHopMajorProgenitorsStrategy(
                    neighbor_halo
                ).all()
                mstar = []
                for prog in progs:
                    prog_halo_num = int(prog.halo_number)
                    prog_timestep = str(self.get_timestep_number(prog))

                    snapshot_name = (
                        pathlib.Path(self.sim_config["SIM_DIR"])
                        .glob("*00" + prog_timestep)
                    )
                    snapshot_name = next(snapshot_name).as_posix()
                    prog_halo = (
                        pynbody.load(snapshot_name)
                        .halos(dosort=True)
                        .load_copy(prog_halo_num)
                    )
                    prog_halo.physical_units()

                    mstar.append(prog_halo.s["mass"].sum())

                try:
                    mstar_max = np.hstack(mstar).max()
                except ValueError:
                    mstar_max = 0.0
            else:
                mstar_max = 0.0

            mstar_max_arr.append(mstar_max)
        return np.asarray(mstar_max_arr)

    def save_interaction_network(self, halo_num, df):
        """Save to disk

        Args:
            halo_num (int): Halo number
            df (pandas.DataFrame): DataFrame containing interaction network
        """
        df.to_hdf(self.sim_config["OUT_FILE"], key="r" + str(halo_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append new properties to the halo interaction network for input Romulus25 halo at step 7779"
    )
    parser.add_argument("--halos", help="File containing halo numbers at step 7779")
    parser.add_argument("--verbose", help="Verbose", action="store_true")
    args = parser.parse_args()

    verbose = bool(args.verbose)
    config = read_json(generate.CONFIG)
    args = parser.parse_args()

    halo_numbers = get_halos(args.halos)
    assert len(halo_numbers) > 0
    assert type(halo_numbers[0]) == int

    if verbose:
        halo_numbers = tqdm.tqdm(halo_numbers)

    din = AddSnapshotProperties(config_filename=generate.CONFIG)

    for halo_number in halo_numbers:
        data = din.query(halo_number)
        din.save_interaction_network(halo_number, data)