from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "easysurfvis",
    version = "0.0.1",
    author = "seojin",
    author_email = "pures1@hanyang.ac.kr",
    description = "visualize surface map easily",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/SeojinYoon/easy_surf_vis",
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires = [
        "numpy",
        "pandas",
    ]
)

