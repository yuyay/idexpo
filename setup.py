from setuptools import setup, find_packages

setup(
    name="idexpo",
    version="0.1",
    packages=find_packages(
        where='src',
        exclude=(
            '.vscode', 'scripts', 'experiments', 'datasets', '.python-version',
            'examples', 'weights', 'data'
        )
    ),
    package_dir={"": "src"},
    install_requires=[
        "numpy", "torch", "torchvision", "torchmetrics", "matplotlib", "pandas", 
        "einops", "pytorch-lightning", "fire", "lightning-bolts", "seaborn"],
)
