===============
Python Learning
===============

This is my playground for Python, learning from the ground up.  I've used python as installed in the OS for systemy type tasks, for CI/CD with Gitlab and Docker, but haven't built too many apps or libraries.  I'm going to try to figure out the best practices and document them here.  Let's start with dependency management.

###########
rst vs md 
###########

**Whoops!  I thought you said "Let's start with dependency management"?**  I did.  I came back an wrote this after the fact when I learned that python project documentation is commonly written in `Link reStructured Text Primer <https://python-docs.readthedocs.io/en/latest/writing/documentation.html#restructuredtext>`_ (rst) as opposed to markdown (md).  The reason for this is `Link Sphinx <http://www.sphinx-doc.org/en/master/>`_, a Python documentation tool, written in Python, which converts rst documents to a variety of formats.

Given that I'm learning Python, I'm going to convert this once md doc into an rst.  If you find any mistakes, you know how they got there.

##################################
The world of dependency management
##################################

`Link PyPA <https://www.pypa.io/en/latest/>`_ - **Python Packaging Authority** -  a working group that maintains many of the `Link relevant projects <https://packaging.python.org/key_projects/>`_ in Python packaging.

[PyPI](https://pypi.org/) - **Python Package Index** - a repository of software for the Python programmning language.  Your company may host its own repository to ensure that only packages with approved licenses are used.  In this case you will need to update your "configuration" (`pip.conf`) to point to your repository.  The configuration file can be created in either `$HOME/.config/.pip/pip.conf` or `$HOME/.pip/pip.conf`.  The file will look somthing like this:   
```
[global]
index = https://YOUR_HOST/repository/pypi.python.org/pypi
index-url = https://YOUR_HOST/repository/pypi.python.org/simple
trusted-host=YOUR_HOST
```


The answer to this [question](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe) from Stackoverflow does a great job of listing and explaining all the relevant choices in the area of dependency management in Python.  

The recommendation from PyPA is to use [pipenv](https://pipenv.readthedocs.io/en/latest/) to manage dependencies. 

## Project Structure

```
README.rst
LICENSE
setup.py
requirements.txt
sample/**init**.py
sample/core.py
sample/helpers.py
docs/conf.py
docs/index.rst
tests/test_basic.py
tests/test_advanced.py
```
