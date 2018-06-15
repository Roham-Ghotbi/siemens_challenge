"""
Setup of Toyota Picking Challenge (tpc) python codebase
Author: Michael Laskey
"""
from setuptools import setup

setup(name='tpc',
      version='0.1.dev0',
      description='project code for the Toyota Picking Challenge',
      author='Michael Laskey',
      author_email='laskeymd@berkeley.edu',
      package_dir={'': 'src'},
      packages=['tpc', 'tpc.config', 'tpc.perception', 'tpc.manipulation', 'tpc.data', 'tpc.offline', 'tpc.detection']
     )
