import os
import sys
from ase.io import read, write
from ase.mep import NEBTools
from ase.visualize import view

images = read("NEB/Fe/098/CO/neb.traj", index=":")


ens = []
for atoms in images:
    ens.append(atoms.get_potential_energy())
print(ens)

nebtools = NEBTools(images)
Ef, dE = nebtools.get_barrier()
max_force = nebtools.get_fmax()
fig = nebtools.plot_band()
fig.savefig('neb_barrier.png')
fig.show()
view(images)



