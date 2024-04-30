#!/bin/sh

# To run DMRGSCF, block2 can be interfaced with pyscf via the dmrgscf extension
# ('pyscf-dmrgscf'). Unfortunately, one has to initialize a settings file
# ('settings.py') in an extra step. The purpose of this script is to create said file.
#
# Note: this script assumes that `pip install` has already been run and pyscf, block2,
# and pyscf-dmrgscf are already installed.

print_usage() {
    echo \
" Script for initializing pyscf-dmrgscf settings.py .

  Options:
  -h, --help     show this menu
  -f, --force    overwrite existing settings.py
"
}

overwrite_settings=false
while [ "${1:-}" != "" ]; do
    case "${1}" in
        "-h" | "--help")
            print_usage
            exit 0
            ;;
        "-f" | "--force")
            overwrite_settings=true
            ;;
        *)
            echo "Error: undefined option '${1}'"
            print_usage
            exit 1
            ;;
    esac
    shift
done

PYSCF_DIR=$(python -c "import pyscf; print(pyscf.__path__[0])")
BLOCK2_EXE=$(which block2main)
DMRGSCF_SETTINGS=${PYSCF_DIR}/dmrgscf/settings.py

if [ ! -d "${PYSCF_DIR}" ] || [ ! -f "${BLOCK2_EXE}" ]; then
    echo "PYSCF_DIR is ${PYSCF_DIR}"
    echo "BLOCK2_EXE is ${BLOCK2_EXE}"
    echo "Invalid pyscf path or block2 path! Exiting..."
    exit 1
fi

if [ -f "$DMRGSCF_SETTINGS" ] && [ "$overwrite_settings" = false ]; then
    echo "${DMRGSCF_SETTINGS} already exists, use '-f' option to overwrite."
    exit 0
fi

echo \
"import os
from pyscf import lib

BLOCKEXE = '${BLOCK2_EXE}'
BLOCKEXE_COMPRESS_NEVPT = BLOCKEXE
# BLOCKSCRATCHDIR = os.path.join('./scratch', str(os.getpid()))
BLOCKSCRATCHDIR = os.path.join(lib.param.TMPDIR, str(os.getpid()))
BLOCKRUNTIMEDIR = '.'
# BLOCKRUNTIMEDIR = str(os.getpid())
MPIPREFIX = '' # change to srun for SLURM job system
BLOCKVERSION = None" \
    > ${DMRGSCF_SETTINGS}

echo "Settings written to ${DMRGSCF_SETTINGS} ."
