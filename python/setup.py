from setuptools import setup, find_packages

with open('README.rst') as f:
  readme = f.read()

with open('LICENSE') as f:
  license = f.read()

setup(
  name='helloworldbj',
  version='0.1.0',
  description='My first structured python module',
  long_description=readme,
  author='B J Floyd',
  author_email='gpanah1927@gmail.com',
  url='https://github.com/gpanah/lernen/python',
  license=license,
  packages=find_packages(exclude=('tests','docs'))

)
