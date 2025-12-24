#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOAP Deduplication Tool Setup
基于 Cos 相似度 和 SOAP 描述符的结构去重工具
"""

from setuptools import setup, find_packages

setup(
    name="COSOAP",
    version="0.1.0",
    author="Ycx",
    description="SOAP-based structure deduplication with labeled data priority",
    packages=find_packages(), 
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires="",
    install_requires=[
        "ase",
        "dscribe",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pytest",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "COSOAP=COSOAP.__main__:main",
        ],
    },
    package_data={
        "COSOAP": ["*.py"],
    },
    include_package_data=True,
    zip_safe=False,
)
