"""Setup configuration for NBA Props Model package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nba_props_model",
    version="0.1.0",
    author="Your Name",
    description="NBA player props prediction model using advanced analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nba_props_model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "selenium>=4.0.0",
        "psutil>=5.9.0",
        "nba-api>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "certifi>=2023.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nba-scraper=nba_props_model.data.scrapers.ctg_scraper:main",
            "nba-pipeline=nba_props_model.pipelines.training_pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)