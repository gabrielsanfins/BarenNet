from setuptools import setup, find_packages

setup(
    name='barennet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.4',
        'pandas>=1.3.4',
        'tensorflow>=2.14.0',
        'tensorflow_cpu>=2.9.1'
    ],
    author="Gabriel Sanfins",
    author_email="gabrielsanfins@gmail.com",
    description="Public repository for research in automatic incomplete similarity discovery",
)
