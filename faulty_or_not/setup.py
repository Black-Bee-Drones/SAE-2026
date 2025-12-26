from setuptools import find_packages, setup
import glob
import os

package_name = 'faulty_or_not'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob.glob('models/*')),
        ('share/' + package_name + '/assets', glob.glob('faulty_or_not/assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryan',
    maintainer_email='controleryan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'faulty_or_not = faulty_or_not.sm:main',
            'test_gauge_publisher = faulty_or_not.test_gauge_publisher:main',
        ],
    },
)
