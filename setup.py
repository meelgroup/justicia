from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
  name = 'justicia',
  packages = ['justicia'],
  version = 'v0.0.1',
  license='MIT',
  description = 'This library can be used formally verify machine learning models on multiple fairness definitions.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Bishwamittra Ghosh',
  author_email = 'bishwamittra.ghosh@gmail.com',
  url = 'https://github.com/meelgroup/justicia',
  download_url = 'https://github.com/meelgroup/justicia/archive/v0.0.1.tar.gz',
  keywords = ['ML fairness verification', 'Application of SSAT solvers'],   # Keywords that define your package best
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)