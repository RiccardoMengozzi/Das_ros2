from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob("launch/aggregative_optimization.launch.py"))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicola',
    maintainer_email='nicola.francesconi@studio.unibo.it',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'agent = agent.agent:main',
        'static_transform = agent.static_transform:main',
        'markers_publisher = agent.markers_publisher:main',
        ],
    },
)
