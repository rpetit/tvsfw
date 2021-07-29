import setuptools

with open("../tvsfw/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tvsfw",
    version="0.0.1",
    author="Romain Petit",
    author_email="romain.petit@inria.fr",
    description="Total (gradient) variation sliding Frank-Wolfe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rpetit/tvsfw",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)