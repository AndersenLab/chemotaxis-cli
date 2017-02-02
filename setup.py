from setuptools import setup
import glob
import os

with open('requirements.txt') as f:
    required = [x for x in f.read().splitlines() if not x.startswith("#")]

from ct import __version__, _program

setup(name=_program,
      version=__version__,
      packages=['ct'],
      description='Count worms from chemotaxis assays',
      url='https://github.com/AndersenLab/chemotaxis-cli',
      author='YOUR NAME',
      author_email='danielecook@gmail.com',
      license='MIT',
      entry_points="""
      [console_scripts]
      {program} = ct.command:main
      """.format(program = _program),
      keywords=[],
      tests_require=['pytest', 'coveralls'],
      zip_safe=False)
