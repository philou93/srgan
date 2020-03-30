import setuptools


setuptools.setup(
    name="SRGAN_1",
    version="0.0.1",
    description="super resolution GAN",
    packages=setuptools.find_packages(),
    install_requires=[
        'keras',
        'h5py',
        'opencv-python',
        'numpy'
    ],
    include_package_data=True
)