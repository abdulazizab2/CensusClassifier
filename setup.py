from setuptools import setup, find_packages

setup(
    name="CensusClassifier",
    version="0.0.1",
    description="CensusClassifier package.",
    author="abdulazizab2",
    packages=find_packages(include=["CensusClassifier", "CensusClassifier.*"]),
    install_requires=[],
)
