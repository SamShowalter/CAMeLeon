#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='cameleon',
      version='1.0',
      description='A simple, flexible interface to apply CAML/CARLI to and train agents on arbitrary RL environments',
      url='https://gitlab.sri.com/caml/cameleon',
      author='Sam Showalter',
      author_email='sam.showalter@sri.com',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          # 'python >= 3.8',
          'gym',
          'gym-minigrid',
          'numpy',
          'matplotlib',
          'tqdm',
          'ray[default]',
          'ray[rllib]',
          'ray[tune]',
          'torch',
          'tensorflow',
          # 'pickle',
          'hickle',
          'argparse',
          'pathlib',
          'datetime',
      ],
      extras_require={
          'caml': [
              'gin-config >= 0.1.1, <= 0.14',
              'absl-py >= 0.2.2',
              'imago @ git+https://gitlab.sri.com/caml/imago',
              'interestingness-xdrl @ git+https://gitlab.sri.com/caml/interestingness-xdrl',
          ],
      },
      zip_safe=True)
