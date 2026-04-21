from setuptools import find_packages, setup


setup(
    name="unified_autonomy",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "setuptools",
        "PyYAML",
        "numpy",
        "scipy",
        "cvxpy",
        "opencv-python",
        "onnxruntime",
        "fastapi",
        "uvicorn",
    ],
    zip_safe=True,
    maintainer="Ariashi",
    maintainer_email="you@example.com",
    description="Unified autonomy platform for classical and learning-based control demos.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "main_demo = unified_autonomy.main_demo:main",
        ],
    },
)
