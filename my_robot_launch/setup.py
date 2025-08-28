from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'bridge'), glob('bridge/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rym',
    maintainer_email='mhadhbirim123@gmail.com',
    description='My robot launch files and actor controller',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'actor_controller = my_robot_launch.actor_controller:main',
        ],
    },
)
