from setuptools import setup, find_packages

setup(
    name='phm_feature_lab',
    version='0.1.0',
    author='Maria Clara Assunção Viana',
    packages=find_packages(include=['phm_feature_lab', 'phm_feature_lab.*']),  # Pacote e subpacotes
    install_requires=[],
)