from setuptools import setup, find_packages

setup(
    name="decision_support_system",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "matplotlib",
        "pyyaml",
        "pytest"
    ],
    entry_points={
        "console_scripts": [
            "decision-support=main:main",
        ],
    },
)
