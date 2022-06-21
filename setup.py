from codecs import open
from os import path

from setuptools import setup

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

the_lib_folder = path.dirname(path.realpath(__file__))
requirementPath = the_lib_folder + '/requirements.txt'
install_requires = []
if path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name="mp3stego-lib",
    version="1.1.6",
    author="Aviad Seady, Tomer Shay, Lee Zaid",
    author_email="aviadevelops@gmail.com, tomershay100@gmail.com, lizizaid@gmail.com",
    description="mp3 steganography library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomershay100/mp3-steganography-lib",
    project_urls={
        "Bug Tracker": "https://github.com/tomershay100/mp3-steganography-lib/issues",
    },
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["mp3stego"],
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=install_requires
)
