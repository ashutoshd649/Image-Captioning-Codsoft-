from setuptools import setup, find_packages

setup(
    name="image_captioning",
    version="1.0.0",
    description="Image Captioning AI — CNN + LSTM / Transformer",
    author="Internship Project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "nltk>=3.8.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
    ],
)
