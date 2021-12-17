from pyscf.gto import Mole
from pyscf.scf import ROHF
from pyscf.mp import MP2
from pyscf.lib import logger
from pyscf.tools import molden

from asf import ASFDMRG

################################################################################
# Singlet oxygen example
################################################################################

# To obtain orbitals more easily, we perform an SCF calculation on triplet oxygen first.
# This avoids convergence problems with singlet oxygen.
mol_triplet = Mole()
mol_triplet.atom = '''
O   0.604   0.000   0.000
O  -0.604   0.000   0.000
'''
mol_triplet.basis = 'def2-SVP'
mol_triplet.spin = 2
mol_triplet.verbose = logger.INFO
mol_triplet.build()

mf_triplet = ROHF(mol_triplet)
mf_triplet.kernel()

# store the triplet ROHF orbitals in a molden file for viewing them later
molden.from_scf(mf_triplet, 'triplet_rohf.molden')

# Singlet oxygen
mol_singlet = mol_triplet.copy()
mol_singlet.spin = 0
mol_singlet.build()

# Now we just take the triplet orbitals as they are and perform a selection for the singlet state.
# 16 is the number of all electrons in O2
# We run the DMRG calculation over the 20 lowest lying orbitals, which is an arbitrary number.
sf = ASFDMRG(mol_singlet, mf_triplet.mo_coeff, 16, 20)
# Run the DMRG calculation.
sf.runDMRG(maxM=200, tol=1e-5)
# Now we select an active space with default settings.
nel, mo_list = sf.selectOrbitals()
