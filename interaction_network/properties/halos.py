import pandas as pd
import tqdm
import tangos as db
import numpy as np
import pathlib
import pynbody
from . import BaseProperties


class HaloProperties(BaseProperties):

    names = ["mvir_max_neighbor"]

    def __init__(self, config_filename=None, verbose=False):
        super().__init__(config_filename=config_filename, verbose=verbose)

    def calculate(self):
        """Generate interaction parameters for given halo

        Args:
            halo_num (int): halo number at desired step
        """
        mvir_max = self._calc_halo_mvir_max(self.data)
        return mvir_max

    def _calc_halo_mvir_max(self, data):
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

    def _calc_halo_mstar_max(self, data):
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

                    snapshot_name = pathlib.Path(self.sim_config["SIM_DIR"]).glob(
                        "*00" + prog_timestep
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