import os
from setuptools import setup, find_packages


def read_requirements():
    """Parses requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='rs_utils',
    version='0.1.0',
    install_requires=read_requirements(),
    license='proprietary',
    description='Utilities for realsense',
    author='takuya-ki',
    author_email='taku8926@gmail.com',
    url='https://takuya-ki.github.io/',
    packages=find_packages(where='rs_utils'),
    package_dir={'': 'rs_utils'},
    python_requires='>=3.7',
)
