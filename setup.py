from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Grapple",
    version="1.0.0",
    description="Runs through a series of Haystack models with parameters defined via yaml",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="robbyzar",
    packages=['Grapple'],
    python_requires=">=3.5",
    install_requires=["pyyaml"]
)
