from setuptools import setup, find_packages

setup(
    name='EZAI',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy<2.0',
        'pandas',
        'torchvision',
        'scikit-learn'
    ],
    description='A simple wrapper library for PyTorch to facilitate easy model training',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/EZAI',  # Replace with your GitHub URL
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)
