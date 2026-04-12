import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "docutils>=0.16",
        "graphviz>=0.14",
        "Grid2Op>=1.12.1",
        "lightsim2grid>=0.10.3",
        "networkx>=2.4",
        "numpy>=1.18.4",
        "pandapower>=2.2.2",
        "scipy>=1.6.0",  # needed for pandapower compatibility for now
        "pandas>=1.0.4",
        "pydot>=1.4.1",
        "matplotlib>=3.3.3",
        "pytest>=4.4.2",
        "Sphinx>=3.1.2",
        "rustworkx>=0.17",
    ],
    "extras": {
        "optional": [
            "oct2py>=5.0.4",
            "pypower>=5.1.4",
            "pypownet>=2.2.0"
        ]
    }
}

setup(name='ExpertOp4Grid',
      version='0.3.0.post1',
      description='Expert analysis algorithm for solving overloads in a powergrid',
      long_description_content_type="text/markdown",
      python_requires=">=3.9",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English",
          "Operating System :: OS Independent",
      ],
      keywords='ML powergrid optmization RL power-systems',
      author='Antoine Marot',
      author_email='antoine.marot@rte-france.com',
      url="https://github.com/marota/ExpertOp4Grid/",
      download_url = 'https://github.com/marota/ExpertOp4Grid/archive/refs/tags/0.1.3.post1.tar.gz',
      license='Mozilla Public License 2.0 (MPL 2.0)',
      packages=setuptools.find_packages(),
      extras_require=pkgs["extras"],
      include_package_data=True,
      install_requires=pkgs["required"],
      zip_safe=False,
      entry_points={'console_scripts': ['expertop4grid=alphaDeesp.main:main']},
)
