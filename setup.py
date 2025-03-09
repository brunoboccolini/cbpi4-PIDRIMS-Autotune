from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='cbpi4-PIDRIMS_AutoTune',
      version='1.0.0',
      description='CraftBeerPi4 RIMS PID AutoTune Plugin',
      author='Bruno Boccolini',
      author_email='bruno.boccolini@gmail.com',
      url='https://github.com/brunoboccolini/cbpi4-PIDRIMS_AutoTune',
      license='GNU General Public License v3 (GPLv3)',
      include_package_data=True,
      package_data={
        # If any package contains *.txt or *.rst files, include them:
      '': ['*.txt', '*.rst', '*.yaml'],
      'cbpi4-PIDRIMS_AutoTune': ['*','*.txt', '*.rst', '*.yaml']},
      packages=['cbpi4-PIDRIMS_AutoTune'],
      long_description=long_description,
      long_description_content_type='text/markdown'
)
