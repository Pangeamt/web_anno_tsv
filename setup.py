import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="web_anno_tsv", # Replace with your own username
    version="0.0.1",
    author="Laurent Bié",
    author_email="l.bie@pangeanic.com",
    description="Read and an write web anno tsv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pangeamt/web_anno_tsv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)