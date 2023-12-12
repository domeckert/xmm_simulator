from setuptools import setup


setup(
      name='xmm_simulator',    # This is the name of your PyPI-package.
      version='0.2.0',
      description='XMM photon simulator from boxes extracted from hydro sims',
      author='Dominique Eckert',
      author_email='Dominique.Eckert@unige.ch',
      #url="https://github.com/domeckert/pyproffit",
      packages=['xmm_simulator'],
      install_requires=[
            'numpy','scipy','astropy','matplotlib','pyatomdb','threeML',
      ],
      package_data={'xmm_simulator': ['rmfs/*', 'imgs/*', 'fwc/*', 'pts/*']},
)

