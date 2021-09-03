#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='interestingness-xdrl',
      version='1.0',
      description='eXplainable Deep Reinforcement Learning (xdrl) framework based on the concept of Interestingness Elements',
      url='https://gitlab.sri.com/caml/interestingness-xdrl',
      author='Pedro Sequeira',
      author_email='pedro.sequeira@sri.com',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'numpy >= 1.13, <= 1.16.4',
          'pandas',
          'jsonpickle',
          'matplotlib',
          'pillow',
          'opencv-python',
          'scikit-video',
          'tqdm'
      ],
      extras_require={
          'caml': [
              'gin-config >= 0.1.1, <= 0.1.4',
              'tensorflow-probability >= 0.4.0, <= 0.6.0',
              'tensorflow == 1.13.2',
              'absl-py >= 0.2.2',
              'pysc2 @ git+https://gitlab.sarnoff.com/jesse.hostetler/pysc2.git@pysc2v3',
              'reaver @ git+https://gitlab.sarnoff.com/jesse.hostetler/reaver.git@caml',
              'sc2recorder @ git+https://gitlab.sri.com/caml/sc2recorder',
              'sc2scenarios @ git+https://gitlab.sri.com/caml/sc2scenarios',
              'imago @ git+https://gitlab.sri.com/caml/imago',
          ],
          'gpu': [
              'tensorflow-gpu == 1.13.2',
          ],
          'reaver': [
              'reaver @ git+https://gitlab.sarnoff.com/jesse.hostetler/reaver.git@caml',
              'gin-config >= 0.1.1, <= 0.1.4',
              'tensorflow-probability >= 0.4.0, <= 0.6.0',
              'tensorflow == 1.13.2',
          ],
          'sc2': [
              'absl-py >= 0.2.2',
              'pysc2 @ git+https://gitlab.sarnoff.com/jesse.hostetler/pysc2.git@pysc2v3',
          ],
          'mac_os': [
              'pyobjc-framework-Quartz'
          ]
      },
      zip_safe=True)
