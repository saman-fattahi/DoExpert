from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="doexpert",
    version="2.0.0",
    author="KSF Institute - Advanced Manufacturing",
    description="Expert Design of Experiments & Process Optimization Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DoExpert",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/DoExpert/issues",
        "Documentation": "https://github.com/yourusername/DoExpert/wiki",
        "Source Code": "https://github.com/yourusername/DoExpert",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Natural Language :: English",
    ],
    python_requires=">=3.11",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pymoo>=0.6.0",
        "SALib>=1.4.0",
        "pyDOE2>=1.3.0",
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "gunicorn>=21.2.0",
        "numba>=0.58.0",
        "Cython>=0.29.0",
        "autograd>=1.6.0",
        "statsmodels>=0.14.0",
        "pingouin>=0.5.3",
        "patsy>=0.5.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.990",
        ],
        "advanced": [
            "pyomo>=6.6.0",
            "dexpy>=0.13",
            "diversipy>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "doexpert=streamlit_app_doexpert:main",
        ],
    },
    include_package_data=True,
    keywords="doe design-of-experiments optimization mcdm streamlit",
    author_email="saman.fattahi@hs-furtwangen.de",
    license="MIT",
)
