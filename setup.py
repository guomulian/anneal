import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anneal",
    version="0.0.1",
    author="Anne Kwok",
    author_email="guomulian@gmail.com",
    description="A template class for implementing simulated annealing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guomulian/anneal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
