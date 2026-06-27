from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cv_detection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ── add this ──────────────────────────────────────────────
        (os.path.join('share', package_name, 'models', 'spearhead'),
            glob('models/spearhead/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luffy',
    maintainer_email='pratusonwalkar06@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'spearhead = cv_detection_pkg.spearhead:main',
        'spearhead_apriltag = cv_detection_pkg.spearhead_apriltag:main',
        'apriltags_2tags = cv_detection_pkg.apriltags_2tags:main',
        'spearhead_distance = cv_detection_pkg.spearhead_distance:main',
        'test = cv_detection_pkg.test:main',
        'depthcam = cv_detection_pkg.depthcam:main',
    ],
},
)
