from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "easysurfvis",
    version = "0.0.1dev5",  # 버전 하나 올림 (dev4 -> dev5)
    author = "seojin",
    author_email = "pures1@hanyang.ac.kr",
    description = "visualize surface map easily",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/SeojinYoon/easy_surf_vis",
    
    # [핵심 변경] src 폴더 내의 모든 패키지(cores 포함)를 찾음
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    
    # [핵심 변경] MANIFEST.in에 정의된 데이터 파일들을 포함시킴
    include_package_data=True,
    
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

