MARIO setup
===========

This folder contains helper files to install and test the MARIO Python package.

Quick install (current environment):

PowerShell:

  python -m pip install --upgrade pip
  pip install mariopy

CMD:

  python -m pip install --upgrade pip
  pip install mariopy

Or with Conda (recommended if you want an isolated env):

  conda create -n mario python=3.12
  conda activate mario
  pip install mariopy

Advanced (install from source):

  git clone https://github.com/it-is-me-mario/MARIO.git
  cd MARIO
  pip install -e .

Files in this folder:

- [install_mario.ps1](install_mario.ps1) — PowerShell installer
- [install_mario.bat](install_mario.bat) — Windows CMD installer
- [requirements.txt](requirements.txt) — package pin for pip
- [mario_test.py](mario_test.py) — quick sanity check script

Documentation: https://mario-suite.readthedocs.io/en/latest/setup/index.html
