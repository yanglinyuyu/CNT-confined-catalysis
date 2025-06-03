from chgnet.model.dynamics import CHGNetCalculator
from ase.io import read
from ase.build import molecule, fcc111
from ase.constraints import FixAtoms
from gcmc import GCMC # from ase.optimize.gcmc import GCMC

###
import torch
from multiprocessing import cpu_count
num = cpu_count()
torch.set_num_threads(num)
######

atoms = fcc111("Pt", size=(4, 4, 4), vacuum=20)
atoms.set_pbc((1, 1, 1))

fix = FixAtoms(indices=[atom.index for atom in atoms if atom.index < 16])
atoms.set_constraint(fix)

ads = [molecule("O")]

calc = CHGNetCalculator()
atoms.calc = calc

gcmc = GCMC(atoms=atoms,
            ads=ads,
            u_ref=[-4.90],
            calc=calc,
            logfile='gcmc.log',
            temperature_K=300,
            opt_fmax=0.1,
            opt_steps=300,
            rmin=1.6,
            rmax=2.0,
            seed=0,
            save_opt=False,
            save_all=False,
            displace=False,
            remove=False,
            add=True,
            height=[-3, 2],
            relax=2,
            forces_rejected=True,
            atoms_too_close=True,
            previous=False,
            from_top=True
            )

gcmc.run(steps=1000)