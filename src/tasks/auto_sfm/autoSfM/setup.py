import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_sfm",
    version="0.0.1",
    author="Chinmay Savadikar",
    author_email="csavadi@ncsu.edu",
    description="AutoSfM package for ease of use in Docker.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/precision-sustainable-ag/autoSfM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
