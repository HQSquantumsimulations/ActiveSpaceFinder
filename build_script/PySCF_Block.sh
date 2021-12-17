#!/usr/bin/env bash
# Script to build PySCF and BLOCK with conda
# NOTE: conda should be properly setup before executing this script
# NOTE: Jmol and xvfb need to be installed manually for using the visualization tools.
# NOTE: if the environment variable WORKDIR is not specified manually the installation
#       will download and build inside the directory from where the script is called
# Settings:
# name of conda environment
CONDA_ENV="HQS_ASF"
# specify PySCF version (only tested with 1.7.6!)
PYSCF_VERSION="1.7.6"
# specify Block and boost versions compatible with ASF
BLOCK_VERSION="f95317b08043b7c531289576d59ad74a6d920741"
BOOST_VERSION="boost_1_76_0" # Latest tested version working with Block
BOOST_SOURCE="https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/${BOOST_VERSION}.tar.gz"
# uncomment next line to specify WORKDIR for build !!! USE ABSOLUTE PATH !!!
# WORKDIR="${HOME}/ASF"
# uncomment the next line to confirm that you read the script before executing it
# I_CONFIRM_READING_THIS_SCRIPT="true"
# check if user read the script
if [ ${I_CONFIRM_READING_THIS_SCRIPT:-"false"} == "true" ]; then
  echo "User has confirmed reading the bash script"
  echo "proceeding with setup of PySCF / Block conda environment"
  echo
else
  echo "Please read the bash script before running it"
  exit
fi
# store current working directory
_DIR=$(pwd)
# set WORKDIR to current directory if not set manually
WORKDIR="${WORKDIR-$(pwd)}"
# create WORKDIR
mkdir -p ${WORKDIR}
echo "Build and installation will be performed in WORKDIR: ${WORKDIR}"
# check if conda has been properly initialized in current shell running the script
if [ $(type -t conda) == 'file' ]; then
  echo
  echo "initializing conda ...";
  eval "$(conda shell.bash hook)";
fi
# create conda environment
conda create -n ${CONDA_ENV} nomkl python=3 cmake=3.19.7 make numpy scipy gcc gxx threadpoolctl openmp openblas h5py=3.1.0 -y
# activate conda environment
conda activate ${CONDA_ENV}
# get path to binutils in conda environment
_PATH=${CONDA_PREFIX}/x86_64-conda-linux-gnu/bin${PATH:+:${PATH}}
# update PATH environment variable for conda environment
conda deactivate
conda env config vars set PATH=${_PATH} --name ${CONDA_ENV}
conda activate ${CONDA_ENV}
# change into WORKDIR
cd ${WORKDIR}
# clone pyscf repository
git clone https://github.com/pyscf/pyscf.git
echo
echo "... done"
echo
echo "compile pyscf ..."
echo
# checkout branch 1.7.6
cd pyscf
git checkout v${PYSCF_VERSION}
if [ ${PYSCF_VERSION} == "1.7.6" ]; then
  # patch release version 1.7.6 with selected commits.
  echo "... PySCF version 1.7.6 detected, cherry-picking some useful patches ... "
  # HQS' native RI-MP2 implementation.
  git cherry-pick aacb49124b0110fe168b1b06598d9bddbfe5288c -m 1
  # Printing fix for ah_level_shift.
  git cherry-pick a52aa5947f3697c7dd50f475bf5aa59a574b46d9 -m 1
  # Cholesky orbitals localization.
  git cherry-pick a704bad4e18294ae62c14cea72de90806e97a7e6 -m 1
fi
# build pyscf core module
cd pyscf/lib
mkdir build
cd build
cmake -DBUILD_LIBXC=1 -DBUILD_XCFUN=1 -DBUILD_LIBCINT=1 ..
make
echo
echo "... done"
# install PySCF
echo "install PySCF..."
cd ${WORKDIR}/pyscf
# activate dmrgscf module
sed -i 's/'\''\*dmrgscf\*'\'', //g' setup.py
# install PySCF using pip
pip install .
# determine site-packages location of PySCF install
PYSCF_HOME="$(python -c 'import site; print(site.getsitepackages()[0])')/pyscf"
echo "PYSCF_HOME: ${PYSCF_HOME}"
conda env config vars set PYSCF_HOME=${PYSCF_HOME}
# re-activate conda to properly set PYSCF_HOME environment variable
conda deactivate
conda activate ${CONDA_ENV}
# install Block (and boost)
cd ${WORKDIR}
# build boost
echo "compiling boost..."
wget ${BOOST_SOURCE} > boost.log 2>&1
tar -xvf $(basename ${BOOST_SOURCE}) >> boost.log 2>&1
rm $(basename ${BOOST_SOURCE})
cd ${BOOST_VERSION}
./bootstrap.sh --with-toolset=gcc --prefix=boost_lib --with-libraries=system,filesystem,serialization --without-icu >> boost.log 2>&1
./b2 link=static install  >> boost.log 2>&1
echo "...Done"
# Install Block
cd ${WORKDIR}
echo "compiling Block..."
git clone https://github.com/sanshar/StackBlock.git
cd StackBlock
git checkout ${BLOCK_VERSION}
sed 's/-Werror//' -i Makefile
make -j 1 \
        BOOSTINCLUDE="${WORKDIR}/${BOOST_VERSION}/boost_lib/include" \
        BOOSTLIB="-L${WORKDIR}/${BOOST_VERSION}/boost_lib/lib -lboost_system -lboost_filesystem -lboost_serialization -lrt" \
        OPENMP=yes \
        LAPACKBLAS="-L${CONDA_PREFIX}/lib/ -lblas -llapack" \
        USE_MPI=no \
        USE_MKL=no

mkdir -p ${PYSCF_HOME}/dmrgscf/block_bin
mv block.spin_adapted ${PYSCF_HOME}/dmrgscf/block_bin/
echo "...Done"
# enabling Block in PySCF
DMRGSETTINGS="${PYSCF_HOME}/dmrgscf/settings.py"
echo "import pyscf" > ${DMRGSETTINGS}
echo "BLOCKEXE = '${PYSCF_HOME}/dmrgscf/block_bin/block.spin_adapted'" >> ${DMRGSETTINGS}
echo "BLOCKEXE_COMPRESS_NEVPT = '${PYSCF_HOME}/dmrgscf/block_bin/block.spin_adapted'" >> ${DMRGSETTINGS}
echo "BLOCKSCRATCHDIR = pyscf.lib.param.TMPDIR" >> ${DMRGSETTINGS}
echo "MPIPREFIX = ''" >> ${DMRGSETTINGS}
echo "BLOCKVERSION = '1.5'" >> ${DMRGSETTINGS}
# cleanup conda 
conda clean --all -y
# deactivate conda environment
conda deactivate
# Final message to user
echo "#######################################################################"
echo "# activate conda environment by running: conda activate ${CONDA_ENV}."
echo "# Remember to install ASF in the ${CONDA_ENV} environment by running"
echo "# 'pip install .' in the main folder of the ASF git repository."
echo "# Please install Jmol and xvfb if you want to use the visualization tools."
echo "# The build and installation process was carried out in WORKDIR:"
echo "# ${WORKDIR}"
echo "# The files generated in this folder *by this script* can be deleted."
echo "#######################################################################"
