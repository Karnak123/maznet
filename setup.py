import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="maznet",
    version="1.0.1",
    description="A lightweight deep learning library",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Karnak123/maznet",
    author="Sagnik Mazumder",
    author_email="sagnikmazumdar37@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["maznet"],
    include_package_data=True,
    install_requires=["numpy"],
)
