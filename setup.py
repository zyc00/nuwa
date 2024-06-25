from setuptools import setup, find_packages

setup(
    name='nuwa',
    version='0.0.93',
    description='',
    author='Xiaoshuai Jet Zhang',
    author_email='jetgabr@gmail.com',
    url='https://github.com/jetd1/nuwa',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'opencv_python>=4.9.0.80',
        'pillow>=10.3.0',
        'matplotlib>=3.8.0',
        'numpy>=1.22',
        'tqdm>=4.0.0',
        'easydict>=1.0',
        'plyfile>=1.0.3',
        'pyransac3d>=0.6.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points=dict(
        console_scripts=[
            "nuwa=nuwa.__main__:main",
            "nuwat=nuwa.__main__:tools",
            "nuwa-tools=nuwa.__main__:tools",
        ]
    )
)
