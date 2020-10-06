from setuptools import find_packages, setup

with open('README.md') as file:
    readme = file.read()

# obtain current version
version = open('asf/__version__.py').read().split()[-1]

requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(name='asf',
      description='An active space finder for CASSCF',
      version=version,
      long_description=readme,
      packages=find_packages(exclude=('docs')),
      author='HQS Quantum Simulations, Reza Shirazi, Thilo Mast',
      author_email='info@quantumsimulations.de',
      url='quantumsimulations.de',
      license=license,
      install_requires=requirements,
      )
