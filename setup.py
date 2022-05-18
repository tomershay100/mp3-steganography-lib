import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    dependencies = list(f)

setuptools.setup(
    name="mp3stego-lib",
    version="0.1.0",
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
    python_requires=">=3.6",
    install_requires=dependencies
)
