import setuptools
from setuptools import find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="sssm",
    version="0.0.6",
    author="chendi",
    author_email="3517725675@qq.com",
    description="a wrapper of sssm",
    long_description_content_type="text/markdown",
    packages=find_packages('sssm'),
    package_dir={"": "sssm"},
    install_requires=[
        'requests',
        'importlib-metadata',
        'torch',
        'pandas',
        'einops',
        'seaborn',
        'numpy',
        'scipy',
        'matplotlib',
'ipywidgets',
'mne'],
)
