import os.path
from setuptools import setup, find_packages
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pandaenv'))
extras = {
"classified_reg": [
      "botorch==0.3.0",
      "pyyaml",
      "hydra-core==0.11.3",
      "nlopt==2.6.2",
      "classireg @ git+ssh://git@github.com/alonrot/classified_regression.git",
            ],
}
extras["all"] = [item for group in extras.values() for item in group]

setup(name='pandaenv',
      version='0.0.1',
      author='Dimitris Gkoutzounis',
      author_email='dimitrios.gkoutzounis@aalto.fi',
      packages=[package for package in find_packages() if package.startswith("pandaenv")],
      package_data={'pandaenv': ['envs/*', 'utils/*']},
      license='MIT',
      extras_require = extras,
      
          classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5'],
)
