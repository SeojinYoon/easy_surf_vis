from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "easysurfvis",
    version = "1.0.2",
    author = "seojin",
    author_email = "pures1@hanyang.ac.kr",
    description = "visualize surface map easily",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/SeojinYoon/easy_surf_vis",
    
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    
    include_package_data=True,
    
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='<=3.12',
    install_requires=[
        "SUITPy",
        "neuroimagingtools",
        "nibabel",
        "nilearn",
        "opencv-python",
        "flask",
        "flask-cors",
    ],
)

