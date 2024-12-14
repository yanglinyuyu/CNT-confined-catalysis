
from ase.optimize import LBFGS
from ase.io import read
from reann import REANN
from ase.visualize import view
from multiprocessing import cpu_count
import torch

### using cpu
num = cpu_count()
torch.set_num_threads(num)


atoms = read("test/Fe30O10-in-CNT.vasp")


"""
atomtype: 
          Fe-CNT-O: ['O', 'C', 'Fe'],
          Fe-CNT-N: ['C', 'N', 'Fe'],
          Fe-CNT-CO: ['O', 'C', 'Fe'],
          Fe-CNT-C: ['C', 'Fe']
"""

atomtype = ['O', 'C', 'Fe']
period = [1, 1, 1]
nn_path = 'Fe-CNT-O.pt'

calc = REANN(atomtype=atomtype,
             period=period,
             nn=nn_path)

atoms.calc = calc
opt = LBFGS(atoms, logfile="-", trajectory="opt.traj")
opt.run(fmax=0.1, steps=500)


# traj = read("opt.traj", index=":")
# view(traj)
