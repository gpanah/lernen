===============
Python Learning
===============

This is my playground for Python, learning from the ground up.  I've used python as installed in the OS for systemy type tasks, for CI/CD with Gitlab and Docker, but haven't built too many apps or libraries.  I'm going to try to figure out the best practices and document them here.  Let's start with basic project structure.

###########
rst vs md 
###########

**Whoops!  I thought you said "Let's start with basic project structure"?**  I did.  I came back an wrote this after the fact when I learned that python project documentation is commonly written in `reStructured Text Primer <https://python-docs.readthedocs.io/en/latest/writing/documentation.html#restructuredtext>`_ (rst) as opposed to markdown (md).  The reason for this is `Sphinx <http://www.sphinx-doc.org/en/master/>`_, a Python documentation tool, written in Python, which converts rst documents to a variety of formats.

Given that I'm learning Python, I'm going to convert this once md doc into an rst.  If you find any mistakes, you know how they got there.

################# 
Project Structure
#################

General Structure

::

  README.rst  
  LICENSE  
  sample/__init___.py
  sample/core.py
  sample/helpers.py
  docs/conf.py
  docs/index.rst
  tests/test_basic.py
  tests/test_advanced.py

Read more `here <https://docs.python-guide.org/writing/structure/>`_.  

From this link, you'll notice that I omitted a couple of files - ``setup.py`` and ``requirements.txt``.  That is because *what* you are building influences which files (and tools) you use for noting and managing dependencies.  

##################################
The world of dependency management
##################################

`PyPA <https://www.pypa.io/en/latest/>`_ - **Python Packaging Authority** -  a working group that maintains many of the `relevant projects <https://packaging.python.org/key_projects/>`_ in Python packaging.

`PyPI <https://pypi.org/>`_ - **Python Package Index** - a repository of software for the Python programmning language.  Your company may host its own repository to ensure that only packages with approved licenses are used.  In this case you will need to update your "configuration" (`pip.conf`) to point to your repository.  The configuration file can be created in either `$HOME/.config/.pip/pip.conf` or `$HOME/.pip/pip.conf`.  The file will look somthing like this:   
::  

  [global]
  index = https://YOUR_HOST/repository/pypi.python.org/pypi
  index-url = https://YOUR_HOST/repository/pypi.python.org/simple
  trusted-host=YOUR_HOST


The answer to this `question <https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe>`_ from Stackoverflow does a great job of listing and explaining all the relevant choices in the area of dependency management in Python.  

The recommendation from PyPA is to use `pipenv <https://pipenv.readthedocs.io/en/latest/>`_ to manage dependencies. *How* that is done, though, gets back to the question above about project structure.  This question is answered nicely by Kenneth Reitz (creator of ``pipenv``) in this `documentation <https://media.readthedocs.org/pdf/pipenv/stable/pipenv.pdf>`_:

(*Section on Pipfile vs setup.py*)

  "There is a subtle but very important distinction to be made between applications and libraries. This is a very common source of confusion in the Python community.  
  
  Libraries provide reusable functionality to other libraries and applications (let’s use the umbrella term projects here). They are required to work alongside other libraries, all with their own set of subdependencies.  They define abstract dependencies.  To avoid version conflicts in subdependencies of different libraries within a project, libraries should never ever pin  dependency versions.  Although they may specify lower or (less frequently) upper bounds, if they rely on some specific feature/fix/bug. Library dependencies are specified via ``install_requires`` in ``setup.py``.

  Libraries are ultimately meant to be used in some application.  Applications are different in that they usually are not depended on by other projects.  They are meant to be deployed into some specific environment and only then should the exact versions of all their dependencies and subdependencies be made concrete.  To make this process easier is currently the main goal of Pipenv."

Reitz goes on to summarize:

  - For libraries, define **abstract dependencies** via ``install_requires`` in ``setup.py``.
  - For applictions, define **dependencies and where to get them** in the *Pipfile* and us this file to update the set of **concrete dependencies** in `Pipfile.lock`.
  - `Pipfile` and Pipenv are useful for developers of libraries to define development or test environments.
  - Where the distinction between library and application isn't that clear, use ``install_requires`` alongside Pipenv and Pipfile.  ``pipenv install -e .`` will tell Pipenv to lock all ``setup.py``-declared dependencies. 
 


