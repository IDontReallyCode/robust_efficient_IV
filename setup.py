import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '4.1.0'
PACKAGE_NAME = 'reiv'
AUTHOR = 'IDontReallyCode'
AUTHOR_EMAIL = 'idontreallycode@outlook.com'
URL = 'https://github.com/IDontReallyCode/baw_iv'

LICENSE = 'MIT'
DESCRIPTION = 'Black-Sholes Implied Volatility, robust and efficient (reiv)'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'numba',
      'scipy'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
