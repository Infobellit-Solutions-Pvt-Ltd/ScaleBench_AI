from setuptools import setup, find_packages

with open("PIP_Package.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scalebench",
    version="0.1.1",
    author="Infobell AI Team",
    author_email="sarthak@infobellit.com",
    description="LLM Inference Benchmarking Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Infobellit-Solutions-Pvt-Ltd/ScaleBench_AI",
    packages=find_packages(include=['scalebench', 'scalebench.*']),
    include_package_data=True,
    package_data={
        'scalebench': ['utils/*.py', '*.py'],
    },
    install_requires=[
        "click",
        "pyyaml",
        "tqdm",
        "pandas",
        "matplotlib",
        "locust",
        "transformers",
        "datasets",
        "tabulate",
        "keyboard",
    ],
    entry_points={
        "console_scripts": [
            "scalebench=scalebench.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)