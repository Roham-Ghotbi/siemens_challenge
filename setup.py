"""
Setup of Deep LfD python codebase
Author: Michael Laskey
"""
from setuptools import setup

setup(name='tpc',
      version='0.1.dev0',
      description='IL HSR project code',
      author='Michael Laskey',
      author_email='laskeymd@berkeley.edu',
      packages=['src/tpc/config', 'src/tpc/perception'],
     )
