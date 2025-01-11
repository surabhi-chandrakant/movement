from setuptools import setup, find_packages

setup(
    name='movement-app',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'flask',
        'tensorflow>=2.12.0',  # Specify compatible versions
        'pygame',
    ],
    entry_points={
        'console_scripts': [
            'movement-cli=app1:main',  # Replace `app1:main` with your actual entry point
        ],
    },
)
