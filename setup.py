from distutils.core import setup
from setuptools import find_packages

setup(
    name='cmbpo',
    packages=find_packages(),
    version='0.0.1',
    description='Constrained Model-based policy optimization',
    long_description=open('./README.md').read(),
    author='Moritz Zanger',
    author_email='zanger.moritz@gmail.com',
    url='https://github.com/anyboby/ConstrainedMBPO/tree/testing/InvVarRollouts',
    entry_points={
        'console_scripts': (
            'cmbpo=softlearning.scripts.console_scripts:main',
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
