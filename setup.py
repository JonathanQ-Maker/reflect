from setuptools import setup, find_packages
import io
import os


with io.open(os.path.join(".", 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name="reflect",
    version="0.0.1",
    description="Numpy based machine learning library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="JonathanQ-Maker",
    author_email="jonathanqfxw@gmail.com",
    url="https://github.com/JonathanQ-Maker/reflect",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=["numpy"],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)