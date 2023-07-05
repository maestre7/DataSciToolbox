from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='DataSciToolbox',
    version='1.0.0',
    author='The Bridge Data Science 2304',
    description='Extensivo repositorio en castellano que abarca todas las fases necesarias para afrontar un proyecto de machine learning, desde el análisis exploratorio de datos hasta la evaluación de los modelos, pasando por las etapas de preprocesamiento, visualización, feature engineering o entrenamiento y evaluación de los modelos.',
    packages=['DataSciToolbox'],
    install_requires=install_requires
)