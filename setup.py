from setuptools import find_packages, setup
import os
# get path to directory containing THIS script
path = os.path.dirname(os.path.abspath(__file__))
# get long description from README
readme = "see README.md"
with open(os.path.join(path, 'README.md')) as readme_file:
    readme = readme_file.read()

# obtain current version
__version__ = None
with open(os.path.join(path, 'asf/__version__.py')) as version_file:
    exec(version_file.read())

# setup requirements
requirements = None
with open(os.path.join(path, 'requirements.txt')) as requirements_file:
    requirements = requirements_file.readlines()
    requirements = [r.strip() for r in requirements]

# get license from LICENSE file (if present)
License = 'Copyright © 2020-2021 HQS Quantum Simulations GmbH. All Rights Reserved.'
with open(os.path.join(path, 'LICENSE')) as license_file:
    License = license_file.read()

setup(
    name='asf',
    description='An active space finder for CASSCF',
    version=__version__,
    long_description=readme,
    packages=find_packages(exclude=('docs')),
    author='HQS Quantum Simulations, Ab-initio team.',
    author_email='info@quantumsimulations.de',
    url='quantumsimulations.de',
    license=License,
    install_requires=requirements,
)
