import copy
import io
import os
import shutil
import time
from math import ceil
import random
import numpy as np
import pandas as pd
from ase.constraints import FixAtoms
from numpy import sqrt
from numpy.random import rand
from ase import Atoms
from ase.io import write, Trajectory, read
from ase.optimize import FIRE2
from ase.optimize.optimize import Optimizer, Dynamics
from ase.units import kB
from ase.build import add_adsorbate
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.ga.utilities import atoms_too_close, closest_distances_generator


class gcmc(Optimizer):
    defaults = {**Optimizer.defaults, 'dr': 0.25, "t": 298 * kB}

    def __init__(self,
                 atoms,
                 ads,
                 calc,
                 u_ref=None,
                 logfile='gcmc.log',
                 label=None,
                 dr=None,
                 temperature_K=None,
                 displace_percentage=0.25,
                 rmin=1.4,
                 rmax=3.0,
                 optimizer=FIRE2,
                 trajectory=None,
                 opt_fmax=None,
                 opt_steps=None,
                 seed=None,
                 save_opt=False,
                 save_all=False,
                 displace=True,
                 remove=True,
                 add=True,
                 magmom=None,
                 relax=0,
                 forces_rejected=False,
                 atoms_too_close=False,
                 previous=False
                 ):
        """Parameters:
        For carbon nanotube confined (or supported) cluster.
        Attention, recommended to use CNTGCMC.

        atoms: ASE Atoms object
            The Atoms object to relax.

        ads: list of Atoms
            The list contains the Atoms object that need to be inserted into the atoms.

        calc: ASE Calculator object
            In order to calculate forces and energies,
            you need to attach a calculator object to your atoms object

        u_ref : List
            The reference chemical potential of adsorbates.

        logfile: string
            Text file used to write summary information.

        optimizer: ASE Optimizer
            Default to use the FIRE.

        temperature_K: float
            temperature, defaults: 298 K

        seed: int
            For a parallel calculation, it is important to use the same seed on all processors!

        relax: int. 0, 1, 2, 3
            0 : no structure relax
            1 : structure optimization was carried out before and after the Metropolis rule.
            2 : only before
            3 : only after

        """
        restart = None
        master = None
        Optimizer.__init__(self, atoms, restart, logfile, master)

        global L, l, nano
        np.random.seed(seed=seed)
        random.seed(a=seed)

        self.traj = []
        self.sa = []
        self.sr = []
        self.all_traj = []
        self.tags_list = []
        self.save_all_step = []
        self.save_all = save_all
        self.atoms = atoms
        self.ads, self.f_ads = self.transfer_ads(ads)
        self.calc = calc
        self.u_ref = u_ref
        self.optimizer = optimizer
        self.save_opt = save_opt
        self.dp = displace_percentage
        self.rmin = rmin
        self.rmax = rmax
        structure = self.get_cluster_position(atoms)
        self.cluster = structure[0]
        self.nano = structure[1]
        self.displace = displace
        self.remove = remove
        self.add = add
        self.ad, self.u = self.get_ad_and_uref()
        self.magmom = magmom
        self.relax = relax
        self.force_rejected = forces_rejected
        self.atoms_close = atoms_too_close
        self.ns = 0

        if self.magmom:
            atoms = self.reset_magmom(atoms, magmom)
            atoms.calc = calc
            self.atoms = atoms

        if label is None:
            self.label = "default"
        elif label == "vasp":
            self.label = "vasp"
            self.vasp = copy.deepcopy(calc)
            if self.relax == 0 or self.relax == 3:
                self.vasp.set(nsw=0)
        else:
            self.label = "default"

        if trajectory is None:
            self.Trajectory = "gcmc.traj"
        else:
            self.Trajectory = trajectory

        if dr is None:
            self.dr = self.defaults['dr']
        else:
            self.dr = dr

        if temperature_K is None:
            self.t = self.defaults['t']
            self.beta = 1 / self.t
            self.temperature = 298
        else:
            self.t = temperature_K * kB
            self.beta = 1 / self.t
            self.temperature = temperature_K

        if opt_fmax is None:
            self.opt_fmax = 0.1
        else:
            self.opt_fmax = opt_fmax

        if opt_steps is None:
            self.opt_steps = 300
        else:
            self.opt_steps = opt_steps

        if previous:
            self.atoms = read(self.Trajectory)
            self.atoms.calc = self.calc
            self.sa = read(self.Trajectory, index=":")
            if os.path.exists(logfile):
                try:
                    data = self.dataframe(logfile)
                    self.nsteps = data["Step"].tolist()[-1]
                    self.ns = data["Step"].tolist()[-1]
                except:
                    self.nsteps = 0
                    self.ns = 0
            else:
                self.nsteps = 0
                self.ns = 0
            self.adsnum = self.get_adsnum(self.atoms)
            max_tag = self.get_atoms_max_tags(self.atoms)
            if max_tag:
                self.tags = max_tag + 1
            else:
                self.tags = 111

            if os.path.exists("all_opt.traj"):
                self.all_traj = read("all_opt.traj", index=":")
            if os.path.exists("all_trial.traj"):
                self.save_all_step = read("all_trial.traj", index=":")
        else:
            tags = self.atoms.get_tags()
            tags = np.unique(tags)
            tags = [int(tag) for tag in tags if tag >= 111]
            if tags:
                max_tag = max(tags)
                self.tags = max_tag + 1
                self.adsnum = len(tags)
            else:
                self.adsnum = 0
                self.tags = 111


        if self.force_rejected and self.relax == 0:
            raise ValueError("In GCMC, force rejected must be associated with structure relax. "
                             "Please set relax = 1, 2 or 3.")

    def initialize(self):
        self._dis = 0
        self._add = 0
        self._rem = 0
        self.trail1 = 0 # displace
        self.trail2 = 0 # remove
        self.trail3 = 0 # add

    def irun(self, steps=None, **kwargs):
        """ call Dynamics.irun and keep track of fmax
        for ase 3.23.0, Dynamics.run and irun must set steps
        """
        self.fmax = 1e-100
        nsteps = steps - self.ns
        return Dynamics.irun(self, steps=nsteps)

    def run(self, steps=None, **kwargs):
        """ call Dynamics.run and keep track of fmax"""
        self.fmax = 1e-100
        nsteps = steps - self.ns
        return Dynamics.run(self, steps=nsteps)

    def dataframe(self, logfile):
        l = []
        reader = io.open(logfile, encoding='utf_8_sig')
        list_data = reader.readlines()
        for i in list_data[2:]:
            l.append(i.split())

        columns = ["Name", "Step", "Date", "Time", "Energy",
                   "fmax", "Temperature", "Trail", "Result",
                   "Adsorbates", "Numbers", "Uads"]

        data = pd.DataFrame(l, columns=columns)

        data["Step"] = pd.to_numeric(data["Step"])
        data["Energy"] = pd.to_numeric(data["Energy"])
        data["fmax"] = pd.to_numeric(data["fmax"])
        data["Date"] = pd.to_datetime(data["Date"])
        data["Time"] = pd.to_datetime(data["Time"])
        data["Numbers"] = pd.to_numeric(data["Numbers"])
        data["Uads"] = pd.to_numeric(data["Uads"])

        return data

    def reset_magmom(self, atoms, magmom):
        atoms = atoms.copy()
        for sym in magmom.keys():
            for atom in atoms:
                if atom.symbol == sym:
                    atom.magmom = magmom.get(sym)
        return atoms

    def transfer_ads(self, ads):
        images = []
        f_images = []
        for ad in ads:
            image = Atoms()
            for atom in ad:
                image.append(atom)
            image.center(vacuum=5)
            f_images.append(image.get_chemical_formula(mode="metal"))
            images.append(image)
        return images, f_images

    def get_ad_and_uref(self, tag=None):
        if isinstance(self.u_ref, list):
            if len(self.ads) == 1:
                self.ad = self.ads[0]
                self.u = self.u_ref[0]
            else:
                if tag is None:
                    self.ad = random.sample(self.ads, 1)[0]
                    for i, a in enumerate(self.ads):
                        if a == self.ad:
                            self.u = self.u_ref[i]
                else:
                    self.ad = Atoms()
                    for atom in self.atoms:
                        if atom.tag == tag:
                            self.ad.append(atom)
                    self.ad.center(vacuum=5)
                    for i, f in enumerate(self.f_ads):
                        f_ad = self.ad.get_chemical_formula(mode="metal")
                        if f_ad == f:
                            self.u = self.u_ref[i]
        else:
            if len(self.ads) == 1:
                self.ad = self.ads[0]
                if self.trail2 == 1:
                    self.u = self.u_ref.get("remove")[0]
                elif self.trail3 == 1:
                    self.u = self.u_ref.get("add")[0]
                else:
                    self.u = self.u_ref.get("add")[0]

            else:
                if tag is None:
                    self.ad = random.sample(self.ads, 1)[0]
                    for i, a in enumerate(self.ads):
                        if a == self.ad:
                            if self.trail2 == 1:
                                self.u = self.u_ref.get("remove")[i]
                            elif self.trail3 == 1:
                                self.u = self.u_ref.get("add")[i]
                            else:
                                self.u = self.u_ref.get("add")[i]
                else:
                    self.ad = Atoms()
                    for atom in self.atoms:
                        if atom.tag == tag:
                            self.ad.append(atom)
                    self.ad.center(vacuum=5)
                    for i, f in enumerate(self.f_ads):
                        f_ad = self.ad.get_chemical_formula(mode="metal")
                        if f_ad == f:
                            if self.trail2 == 1:
                                self.u = self.u_ref.get("remove")[i]
                            elif self.trail3 == 1:
                                self.u = self.u_ref.get("add")[i]
                            else:
                                self.u = self.u_ref.get("add")[i]

        return self.ad, self.u

    def compare_atoms(self, a, b):
        com_a = a.copy()
        com_b = b.copy()
        com = SymmetryEquivalenceCheck()
        compare = com.compare(com_a, com_b)
        return compare

    def compare_close(self, atoms):
        atom_numbers = atoms.get_atomic_numbers()
        atom_numbers = np.unique(atom_numbers)
        bl = closest_distances_generator(atom_numbers=atom_numbers,
                                         ratio_of_covalent_radii=0.6)
        close = atoms_too_close(atoms, bl)
        return close

    def get_traj(self, traj):
        Traj = []
        images = Trajectory(traj)
        Traj.extend(images)
        return Traj

    def get_atoms_max_tags(self, atoms):
        atoms = atoms.copy()
        tags = atoms.get_tags()
        tags = np.unique(tags)
        tags = [int(tag) for tag in tags if tag >= 111]
        if tags:
            max_tag = max(tags)
        else:
            max_tag = None

        return max_tag

    def get_adsnum(self, atoms):
        atoms = atoms.copy()
        tags = atoms.get_tags()
        tags = np.unique(tags)
        tags = [tag for tag in tags if tag >= 111]
        num = len(tags)
        return num

    def remove_dir(self, path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except:
                pass
        else:
            pass

    def save_all_traj(self):
        if self.nsteps == self.max_steps - 1:
            if self.save_opt:
                write("all_opt.traj", self.all_traj)
                path = os.getcwd() + "/opt_traj/"
                self.remove_dir(path)
            if self.save_all:
                write("all_trial.traj", self.save_all_step)

    def structure_opt(self, atoms):
        if self.label == "default":
            if self.save_opt:
                path = os.getcwd() + "/opt_traj/"
                self.create_file(path)
                dyn = self.optimizer(atoms, logfile=None,
                                     trajectory=path + "step_%d_opt.traj" % (self.nsteps + 1))
                dyn.run(fmax=self.opt_fmax, steps=self.opt_steps)
                traj = path + "step_%d_opt.traj" % (self.nsteps + 1)
                Traj = self.get_traj(traj)
                self.all_traj.extend(Traj)
                return atoms
            else:
                dyn = self.optimizer(atoms, logfile=None)
                dyn.run(fmax=self.opt_fmax, steps=self.opt_steps)
                return atoms
        else:
            if self.save_opt:
                path = os.getcwd() + "/opt_traj/"
                self.create_file(path)
                atoms.calc = self.calc
                atoms.get_potential_energy()
                images = read("OUTCAR", index=":")
                write(path + "step_%d_opt.traj" % (self.nsteps + 1), images)
                self.all_traj.extend(images)
                return atoms
            else:
                atoms.calc = self.calc
                atoms.get_potential_energy()
                return atoms

    def create_file(self, path):
        if os.path.exists(path):
            pass
        else:
            try:
                os.makedirs(path)
            except:
                pass

    def get_tag(self, atoms):
        atoms = atoms.copy()
        tags = atoms.get_tags()
        tags = np.unique(tags)
        tags = [tag for tag in tags if tag >= 111]
        return tags


    def get_cluster_position(self, atoms):
        cluster_index = []
        nano_index = []
        for atom in atoms:
            if atom.symbol != "C":
                cluster_index.append(atom.index)
            else:
                nano_index.append(atom.index)
        return cluster_index, nano_index

    def add_judgement(self, atoms):

        exist = []
        r_exist = []
        ads_exist = []
        indexes = []
        cluster = []

        tag = atoms[-1].tag
        for atom in atoms:
            if atom.tag == tag:
                indexes.append(atom.index)

        for index in indexes:
            for atom in atoms:
                if atom.tag != tag:
                    d = atoms.get_distances(index, indices=[atom.index], mic=True)
                    if self.rmin < d < self.rmax:
                        exist.append(atom)
                    if self.rmin > d:
                        r_exist.append(atom)

        for atom in exist:
            if atom.tag >= 111:
                ads_exist.append(atom)
            if atom.index in self.cluster:
                cluster.append(atom)

        if cluster and not ads_exist and not r_exist:
            add = True
        else:
            add = False

        if self.atoms_close:
            close = self.compare_close(atoms)
            if add and not close:
                add = True
            else:
                add = False
        return add

    def move_atom(self, L=None):
        atoms = self.atoms.copy()
        image = self.atoms.copy()

        if L is None:
            L = []
        for atom in atoms:
            L.append(atom.index)

        l = np.random.choice(L, size=ceil((len(L) * self.dp)), replace=False)

        for index in l:
            atoms[index].position = atoms[index].position + self.dr * np.random.uniform(-1., 1., (1, 3))

        rn = atoms.get_positions(wrap=True)
        image.set_positions(rn)
        image.wrap()
        return image

    def add_particle(self, ads, l=None, nano=None):
        atoms = self.atoms.copy()
        ads = ads.copy()
        ads.set_tags(tags=self.tags)
        if l is None:
            l = []
            nano = []
        for atom in atoms:
            if atom.index in self.cluster:
                l.append(atom.position)
            if atom.index in self.nano:
                nano.append(atom.position)
        Min = np.min(l, axis=0)
        Max = np.max(l, axis=0)
        nanomax = np.max(nano, axis=0)
        x = np.random.uniform(Min[0] - 1.5, Max[0] + 1.5)
        y = np.random.uniform(Min[1] - 1.5, Max[1] + 1.5)
        z = np.random.uniform(-(nanomax[2] - Max[2] - 1.5),
                              -(nanomax[2] - Min[2] + 1.5))
        if z < (-(nanomax[2] - Max[2] - 1.5) - (nanomax[2] - Min[2] + 1.5)) / 2.0:
            ads.rotate(a=180, v="x")
        add_adsorbate(atoms, ads, height=z, position=(x, y))
        self.tags += 1

        return atoms

    def remove_particle(self, tag=None):
        atoms = self.atoms.copy()
        image = self.atoms.copy()
        if tag:
            for atom in atoms:
                if atom.tag == tag:
                    index2remove = atom.index
                    del image[[index2remove]]
            constraints = atoms.constraints
            if constraints:
                new_con = []
                fixed_atom_indices = []
                for con in constraints:
                    if isinstance(con, FixAtoms):
                        fixed_atom_indices.extend(con.index)
                    else:
                        new_con.append(con)

                if index2remove in fixed_atom_indices:
                    return atoms
                else:
                    new_fixed_atom_indices = []
                    for i in fixed_atom_indices:
                        if i > index2remove:
                            new_fixed_atom_indices.append(i - 1)
                        else:
                            new_fixed_atom_indices.append(i)

                    fix_constraint = FixAtoms(new_fixed_atom_indices)
                    new_con.append(fix_constraint)

                image.set_constraint(new_con)

            return image
        else:
            return atoms

    def trail_displace(self):
        self.trail1 = 1
        self.trail2 = 0
        self.trail3 = 0
        self._dis = 0
        Eo = self.atoms.get_potential_energy()
        new = self.move_atom()
        compare = self.compare_atoms(self.atoms, new)
        close = self.compare_close(new)
        if not compare and not close:
            if self.magmom:
                new = self.reset_magmom(new, self.magmom)
            En, fmax, new = self.get_energy(new)
            if self.relax == 2 and self.force_rejected and fmax <= self.opt_fmax:
                if En < Eo:
                    self.atoms = new
                    self._dis = 1
                    self.save_accepted_atoms(new)
                elif rand() < np.exp(-(En - Eo) * self.beta):
                    self.atoms = new
                    self._dis = 1
                    self.save_accepted_atoms(new)
                else:
                    self._dis = 0
            else:
                if En < Eo:
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)
                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._dis = 1
                            self.save_accepted_atoms(new)
                        else:
                            self._dis = 0
                    else:
                        self.atoms = new
                        self._dis = 1
                        self.save_accepted_atoms(new)

                elif rand() < np.exp(-(En - Eo) * self.beta):
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)
                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._dis = 1
                            self.save_accepted_atoms(new)
                        else:
                            self._dis = 0
                    else:
                        self.atoms = new
                        self._dis = 1
                        self.save_accepted_atoms(new)
                else:
                    self._dis = 0

        return self.atoms

    def trail_add(self):
        self.trail1 = 0
        self.trail2 = 0
        self.trail3 = 1
        self._add = 0
        Eo = self.atoms.get_potential_energy()
        self.ad, self.u = self.get_ad_and_uref()
        new = self.add_particle(ads=self.ad)
        add = self.add_judgement(new)
        if self.magmom:
            new = self.reset_magmom(new, self.magmom)
        if add:
            En, fmax, new = self.get_energy(new)
            if self.relax == 2 and self.force_rejected and fmax <= self.opt_fmax:
                if En - Eo - self.u < 0:
                    self.atoms = new
                    self._add = 1
                    self.adsnum = self.get_adsnum(self.atoms)
                    self.save_accepted_atoms(new)
                elif rand() < np.exp(-(En - Eo - self.u) * self.beta):
                    self.atoms = new
                    self._add = 1
                    self.adsnum = self.get_adsnum(self.atoms)
                    self.save_accepted_atoms(new)
                else:
                    self._add = 0
            else:
                if En - Eo - self.u < 0:
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)

                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._add = 1
                            self.adsnum = self.get_adsnum(self.atoms)
                            self.save_accepted_atoms(new)
                        else:
                            self._add = 0
                    else:
                        self.atoms = new
                        self._add = 1
                        self.adsnum = self.get_adsnum(self.atoms)
                        self.save_accepted_atoms(new)

                elif rand() < np.exp(-(En - Eo - self.u) * self.beta):
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)

                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._add = 1
                            self.adsnum = self.get_adsnum(self.atoms)
                            self.save_accepted_atoms(new)
                        else:
                            self._add = 0
                    else:
                        self.atoms = new
                        self._add = 1
                        self.adsnum = self.get_adsnum(self.atoms)
                        self.save_accepted_atoms(new)

                else:
                    self._add = 0

        return self.atoms

    def trail_remove(self):
        self.trail1 = 0
        self.trail2 = 1
        self.trail3 = 0
        self._rem = 0
        Eo = self.atoms.get_potential_energy()
        tags = self.get_tag(self.atoms)
        if tags:
            tag = np.random.choice(tags)
        else:
            tag = None
        self.ad, self.u = self.get_ad_and_uref(tag)
        new = self.remove_particle(tag=tag)
        compare = self.compare_atoms(self.atoms, new)
        if not compare:
            if self.magmom:
                new = self.reset_magmom(new, self.magmom)
            En, fmax, new = self.get_energy(new)

            if self.relax == 2 and self.force_rejected and fmax <= self.opt_fmax:
                if En - Eo + self.u < 0:
                    self.atoms = new
                    self._rem = 1
                    self.adsnum = self.get_adsnum(self.atoms)
                    self.save_accepted_atoms(new)
                elif rand() < np.exp(-(En - Eo + self.u) * self.beta):
                    self.atoms = new
                    self._rem = 1
                    self.adsnum = self.get_adsnum(self.atoms)
                    self.save_accepted_atoms(new)
                else:
                    self._rem = 0
            else:
                if En - Eo + self.u < 0:
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)

                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._rem = 1
                            self.adsnum = self.get_adsnum(self.atoms)
                            self.save_accepted_atoms(new)
                        else:
                            self._rem = 0
                    else:
                        self.atoms = new
                        self._rem = 1
                        self.adsnum = self.get_adsnum(self.atoms)
                        self.save_accepted_atoms(new)
                elif rand() < np.exp(-(En - Eo + self.u) * self.beta):
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)

                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._rem = 1
                            self.adsnum = self.get_adsnum(self.atoms)
                            self.save_accepted_atoms(new)
                        else:
                            self._rem = 0
                    else:
                        self.atoms = new
                        self._rem = 1
                        self.adsnum = self.get_adsnum(self.atoms)
                        self.save_accepted_atoms(new)
                else:
                    self._rem = 0

        return self.atoms

    def save_accepted_atoms(self, atoms):
        image = atoms.copy()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        image.calc = sp(atoms=atoms, energy=energy, forces=forces)
        self.sa.append(image)
        write(self.Trajectory, self.sa)

    def get_energy(self, atoms):
        if self.label == "default":
            calc = self.calc
            if self.relax == 1 or self.relax == 2:
                atoms.calc = calc
                if self.save_opt:
                    path = os.getcwd() + "/opt_traj/"
                    self.create_file(path)
                    dyn = self.optimizer(atoms, logfile=None,
                                         trajectory=path + "step_%d_opt.traj" % (self.nsteps + 1))
                    dyn.run(fmax=self.opt_fmax, steps=self.opt_steps)
                    traj = path + "step_%d_opt.traj" % (self.nsteps + 1)
                    Traj = self.get_traj(traj)
                    self.all_traj.extend(Traj)
                else:
                    dyn = self.optimizer(atoms, logfile=None)
                    dyn.run(fmax=self.opt_fmax, steps=self.opt_steps)
            else:
                atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            fmax = sqrt((forces ** 2).sum(axis=1).max())
            if self.save_all:
                image = atoms.copy()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                image.calc = sp(atoms=atoms, energy=energy, forces=forces)
                self.save_all_step.append(image)
        else:
            atoms.calc = self.vasp
            atoms.get_potential_energy()
            if self.save_opt:
                path = os.getcwd() + "/opt_traj/"
                self.create_file(path)
                images = read("OUTCAR", index=":")
                write(path + "step_%d_opt.traj" % (self.nsteps + 1), images)
                self.all_traj.extend(images)

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            fmax = sqrt((forces ** 2).sum(axis=1).max())
            if self.save_all:
                image = read("OUTCAR")
                self.save_all_step.append(image)
        return energy, fmax, atoms

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy()
        if fmax > self.opt_fmax and self.nsteps == 0:
            self.atoms = self.structure_opt(self.atoms)
            forces = self.atoms.get_forces()
            fmax = sqrt((forces ** 2).sum(axis=1).max())
            e = self.atoms.get_potential_energy()
        T = time.localtime()
        if self.logfile is not None:
            trail = "initial"
            result = "initial"
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Date", "Time", "Energy",
                        "fmax", "Temperature", "Trail", "Result",
                        "Adsorbates", "Numbers", "Uref",)
                msg = "%s  %4s %10s %8s %15s %12s %19s %12s %12s %15s %12s %10s\n" % args
                self.logfile.write(msg)
                self.save_accepted_atoms(self.atoms)

            if self.trail1 == 1:
                if self._dis == 1:
                    trail = "displace"
                    result = "accepted"
                else:
                    trail = "displace"
                    result = "refused"
            if self.trail2 == 1:
                if self._rem == 1:
                    trail = "remove"
                    result = "accepted"
                else:
                    trail = "remove"
                    result = "refused"
            if self.trail3 == 1:
                if self._add == 1:
                    trail = "add"
                    result = "accepted"
                else:
                    trail = "add"
                    result = "refused"

            args = (name, self.nsteps, T[0], T[1], T[2], T[3], T[4], T[5],
                    e, fmax, self.temperature, trail, result,
                    self.ad.symbols, self.adsnum, self.u)
            msg = "%s:  %3d %04d/%02d/%02d %02d:%02d:%02d %15.6f %12.4f %18dK" \
                  " %12s %12s %15s %12d %10.6f\n" % args
            self.logfile.write(msg)
            self.logfile.flush()

    def step(self):

        if self.displace and not self.add and not self.remove:
            self.atoms = self.trail_displace()
        elif self.displace and self.add and not self.remove:
            roll = rand()
            if roll < 0.5:
                self.atoms = self.trail_displace()
            else:
                self.atoms = self.trail_add()
        elif self.displace and self.add and self.remove:
            roll = rand()
            if roll < 1.0/3.0:
                self.atoms = self.trail_displace()
            elif roll < 2.0/3.0:
                self.atoms = self.trail_add()
            else:
                self.atoms = self.trail_remove()
        elif self.add and not self.displace and not self.remove:
            self.atoms = self.trail_add()
        elif self.add and self.remove and not self.displace:
            roll = rand()
            if roll < 0.5:
                self.atoms = self.trail_add()
            else:
                self.atoms = self.trail_remove()
        elif self.displace and self.remove and not self.add:
            roll = rand()
            if roll < 0.5:
                self.atoms = self.trail_displace()
            else:
                self.atoms = self.trail_remove()
        elif self.remove and not self.displace and not self.add:
            self.trail_remove()

        self.save_all_traj()
        self.dump((self.atoms))


class GCMC(gcmc):
    defaults = {**Optimizer.defaults, 'dr': 0.25, "t": 298 * kB}

    def __init__(self,
                 atoms,
                 ads,
                 calc,
                 u_ref=None,
                 label=None,
                 logfile='gcmc.log',
                 dr=None,
                 displace_percentage=0.25,
                 temperature_K=None,
                 optimizer=FIRE2,
                 opt_fmax=None,
                 opt_steps=None,
                 rmin=1.2,
                 rmax=3.0,
                 trajectory=None,
                 seed=None,
                 save_opt=False,
                 save_all=False,
                 displace=True,
                 add=True,
                 remove=True,
                 magmom=None,
                 height=None,
                 relax=0,
                 forces_rejected=False,
                 atoms_too_close=False,
                 previous=False,
                 from_top=True
                 ):
        """Parameters:

        For Slab or Surface.

        atoms: ASE Atoms object
            The Atoms object to relax.

        ads: list of ASE Atoms object
            The list contains the Atoms object that need to be inserted into the atoms.

        calc: ASE Calculator object
            In order to calculate forces and energies,
            you need to attach a calculator object to your atoms object

        u_ref : List
            The reference chemical potential of adsorbates.

        logfile: string
            Text file used to write summary information.

        optimizer: ASE Optimizer
            Default to use the FIRE.

        temperature_K: int or float
            temperature, defaults: 298K

        seed: int
            For a parallel calculation, it is important to use the same seed on all processors!

        relax: int. 0, 1, 2, 3
            0 : no structure relax
            1 : structure optimization was carried out before and after the Metropolis rule.
            2 : only before
            3 : only after

        """
        restart = None
        master = None
        Optimizer.__init__(self, atoms, restart, logfile, master)

        global L, l, nano
        np.random.seed(seed=seed)
        random.seed(a=seed)

        self.traj = []
        self.sa = []
        self.sr = []
        self.all_traj = []
        self.tags_list = []
        self.save_all_step = []
        self.save_all = save_all
        self.atoms = atoms
        self.ads, self.f_ads = self.transfer_ads(ads)
        self.calc = calc
        self.u_ref = u_ref
        self.optimizer = optimizer
        self.save_opt = save_opt
        self.dp = displace_percentage
        self.rmin = rmin
        self.rmax = rmax
        self.displace = displace
        self.remove = remove
        self.add = add
        surface_position = self.get_surface_position(atoms)
        self.SFMin = surface_position[0]
        self.SFMax = surface_position[1]
        self.ad, self.u = self.get_ad_and_uref()
        self.magmom = magmom
        self.hight = height
        self.relax = relax
        self.force_rejected = forces_rejected
        self.atoms_close = atoms_too_close
        self.from_top = from_top
        self.ns = 0

        if self.magmom:
            atoms = self.reset_magmom(atoms, magmom)
            atoms.calc = calc
            self.atoms = atoms

        if label is None:
            self.label = "default"
        elif label == "vasp":
            self.label = "vasp"
            self.vasp = copy.deepcopy(calc)
            if self.relax == 0 or self.relax == 3:
                self.vasp.set(nsw=0)
        else:
            self.label = "default"

        if trajectory is None:
            self.Trajectory = "gcmc.traj"
        else:
            self.Trajectory = trajectory

        if dr is None:
            self.dr = self.defaults['dr']
        else:
            self.dr = dr

        if temperature_K is None:
            self.t = self.defaults['t']
            self.beta = 1 / self.t
            self.temperature = 298
        else:
            self.t = temperature_K * kB
            self.beta = 1 / self.t
            self.temperature = temperature_K

        if opt_fmax is None:
            self.opt_fmax = 0.1
        else:
            self.opt_fmax = opt_fmax

        if opt_steps is None:
            self.opt_steps = 300
        else:
            self.opt_steps = opt_steps

        if previous:
            self.atoms = read(self.Trajectory)
            self.atoms.calc = self.calc
            self.sa = read(self.Trajectory, index=":")
            if os.path.exists(logfile):
                try:
                    data = self.dataframe(logfile)
                    self.nsteps = data["Step"].tolist()[-1]
                    self.ns = data["Step"].tolist()[-1]
                except:
                    self.nsteps = 0
                    self.ns = 0
            else:
                self.nsteps = 0
                self.ns = 0
            self.adsnum = self.get_adsnum(self.atoms)
            max_tag = self.get_atoms_max_tags(self.atoms)
            if max_tag:
                self.tags = max_tag + 1
            else:
                self.tags = 111
            if os.path.exists("all_opt.traj"):
                self.all_traj = read("all_opt.traj", index=":")
            if os.path.exists("all_trial.traj"):
                self.save_all_step = read("all_trial.traj", index=":")
        else:
            tags = self.atoms.get_tags()
            tags = np.unique(tags)
            tags = [int(tag) for tag in tags if tag >= 111]
            if tags:
                max_tag = max(tags)
                self.tags = max_tag + 1
                self.adsnum = len(tags)
            else:
                self.adsnum = 0
                self.tags = 111

        if self.force_rejected and self.relax == 0:
            raise ValueError("In GCMC, force rejected must be associated with structure relax. "
                             "Please set relax = 1, 2 or 3.")

    def move_atom(self, L=None):
        atoms = self.atoms.copy()
        image = self.atoms.copy()
        if L is None:
            L = []

        for atom in atoms:
            L.append(atom.index)

        l = np.random.choice(L, size=ceil((len(L) * self.dp)), replace=False)
        for index in l:
            atoms[index].position = atoms[index].position + self.dr * np.random.uniform(-1., 1., (1, 3))

        rn = atoms.get_positions(wrap=True)
        image.set_positions(rn)
        image.wrap()
        return image

    def add_particle(self, ads, **kwargs):
        atoms = self.atoms.copy()
        ads = ads.copy()
        ads.set_tags(self.tags)
        x = np.random.uniform(self.SFMin[0], self.SFMax[0])
        y = np.random.uniform(self.SFMin[1], self.SFMax[1])
        if self.hight:
            z = np.random.uniform(self.hight[0], self.hight[1])
        else:
            z = np.random.uniform(-3, 3.0)
        add_adsorbate(atoms, ads, height=z, position=(x, y))
        self.tags += 1

        return atoms

    def get_surface_position(self, atoms):
        surface_position = atoms.get_positions()
        sur_min = np.min(surface_position, axis=0)
        sur_max = np.max(surface_position, axis=0)
        return sur_min, sur_max

    def add_judgement(self, atoms):

        exist = []
        r_exist = []
        ads_exist = []
        indexes = []
        tag = atoms[-1].tag
        for atom in atoms:
            if atom.tag == tag:
                indexes.append(atom.index)
        for index in indexes:
            for atom in atoms:
                if atom.tag != tag:
                    d = atoms.get_distances(index, indices=[atom.index], mic=True)
                    if self.rmin < d < self.rmax:
                        exist.append(atom)
                    if self.rmin > d:
                        r_exist.append(atom)
        for atom in exist:
            if atom.tag >= 111:
                ads_exist.append(atom)

        if exist and not ads_exist and not r_exist:
            add = True
        else:
            add = False

        if self.atoms_close:
            close = self.compare_close(atoms)
            if add and not close:
                add = True
            else:
                add = False

        return add

    def from_top_tags(self, atoms: Atoms):
        atoms = atoms.copy()
        z_position = [atom.position[2] for atom in atoms if atom.tag >= 111]
        max_z = max(z_position)
        tags = []
        for atom in atoms:
            if atom.tag >= 111 and atom.position[2] >= (max_z - 1.5):
                tags.append(atom.tag)
        tag = np.random.choice(tags)
        return tag

    def trail_remove(self):
        self.trail1 = 0
        self.trail2 = 1
        self.trail3 = 0
        self._rem = 0
        Eo = self.atoms.get_potential_energy()
        tags = self.get_tag(self.atoms)
        if tags:
            if self.from_top:
                tag = self.from_top_tags(self.atoms)
            else:
                tag = np.random.choice(tags)
        else:
            tag = None
        self.ad, self.u = self.get_ad_and_uref(tag)
        new = self.remove_particle(tag=tag)
        compare = self.compare_atoms(self.atoms, new)
        if not compare:
            if self.magmom:
                new = self.reset_magmom(new, self.magmom)
            En, fmax, new = self.get_energy(new)
            if self.relax == 2 and self.force_rejected and fmax <= self.opt_fmax:
                if En - Eo + self.u < 0:
                    self.atoms = new
                    self._rem = 1
                    self.adsnum = self.get_adsnum(self.atoms)
                    self.save_accepted_atoms(new)
                elif rand() < np.exp(-(En - Eo + self.u) * self.beta):
                    self.atoms = new
                    self._rem = 1
                    self.adsnum = self.get_adsnum(self.atoms)
                    self.save_accepted_atoms(new)
                else:
                    self._rem = 0
            else:
                if En - Eo + self.u < 0:
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)

                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._rem = 1
                            self.adsnum = self.get_adsnum(self.atoms)
                            self.save_accepted_atoms(new)
                        else:
                            self._rem = 0
                    else:
                        self.atoms = new
                        self._rem = 1
                        self.adsnum = self.get_adsnum(self.atoms)
                        self.save_accepted_atoms(new)
                elif rand() < np.exp(-(En - Eo + self.u) * self.beta):
                    if self.relax == 1 or self.relax == 3:
                        new = self.structure_opt(new)

                    if self.force_rejected:
                        forces = new.get_forces()
                        fmax = sqrt((forces ** 2).sum(axis=1).max())
                        if fmax <= self.opt_fmax:
                            self.atoms = new
                            self._rem = 1
                            self.adsnum = self.get_adsnum(self.atoms)
                            self.save_accepted_atoms(new)
                        else:
                            self._rem = 0
                    else:
                        self.atoms = new
                        self._rem = 1
                        self.adsnum = self.get_adsnum(self.atoms)
                        self.save_accepted_atoms(new)
                else:
                    self._rem = 0

        return self.atoms



