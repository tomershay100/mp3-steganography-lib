import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mp3stego-lib",
    version="0.0.13",
    author="Seady, Shay, Zaid",
    author_email="tomershay100@gmail.com",
    description="A mp3 decode and encode library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomershay100/MP3-Steganography",
    project_urls={
        "Bug Tracker": "https://github.com/tomershay100/MP3-Steganography/issues",
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
)
