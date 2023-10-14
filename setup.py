from setuptools import setup, find_packages

setup(
    name='barennet',
    version='0.1',
    packages=find_packages(),
    author="Gabriel Sanfins",
    author_email="gabrielsanfins@gmail.com",
    description="Public repository for research in automatic incomplete similarity discovery",
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow_cpu"
    ]
)
