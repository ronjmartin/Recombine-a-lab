from setuptools import setup, find_packages

setup(
    name="pdf-analysis-dashboard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.44.0",
        "pandas>=2.2.0",
        "plotly>=6.0.0", 
        "numpy>=1.26.0"
    ],
    # Include additional files into the package
    package_data={
        # If any package contains .html or .css files, include them:
        "": ["*.html", "*.css", "*.toml"],
    },
    # Entry points for deployments
    entry_points={
        "console_scripts": [
            "run-app=app:main",
        ],
    },
    python_requires=">=3.8",
)