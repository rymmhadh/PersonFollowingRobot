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
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rym Mhadbi',
    maintainer_email='mhadhbirim123@gmail.com',
    description='Launch files and actor controller for my robot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'actor_controller = my_robot_launch.actor_controller:main',
        ],
    },
)
