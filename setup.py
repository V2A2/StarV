from setuptools import setup, find_packages

def read_requirements(filename='requirements.txt'):
    with open(filename) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith('#')]

setup(
    name="starv_dev_test", 
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.6',
)