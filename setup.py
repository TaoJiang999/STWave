from setuptools import Command, find_packages, setup

__lib_name__ = "stwave"
__lib_version__ = "1.0.0"
__description__ = "STWave"
__url__ = "https://github.com/TaoJiang999/STWave"
__license__ = "MIT"
__author__ = "Tao Jiang"
__author_email__ = "taoj@mails.cqjtu.edu.cn"
setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author=__author__,
    author_email=__author_email__,
    license = __license__,
    packages = ['STWave'],
    install_requires = ["requests"],
    zip_safe = False,
    include_package_data = False
)