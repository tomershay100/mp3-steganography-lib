import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

import os

the_lib_folder = os.path.dirname(os.path.realpath(__file__))
requirementPath = the_lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="mp3stego-lib",
    version="0.2.1",
    author="Aviad Seady, Tomer Shay, Lee Zaid",
    author_email="aviadevelops@gmail.com, tomershay100@gmail.com, lizizaid@gmail.com",
    description="A mp3 decode and encode library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomershay100/mp3-steganography-lib",
    project_urls={
        "Bug Tracker": "https://github.com/tomershay100/mp3-steganography-lib/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=(
        setuptools.find_packages(where="src")
    ),
    python_requires=">=3.9",
    install_requires=install_requires
)
