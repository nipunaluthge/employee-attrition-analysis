from setuptools import setup, find_packages

setup(
    name="employee-attrition-analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'xgboost',
        'matplotlib',
        'shap',
    ],
) 