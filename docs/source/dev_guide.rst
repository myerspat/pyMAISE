=========
Dev Guide
=========

----------------------
Installation and Setup
----------------------

Prior to cloning the repository, git and Python must be installed on your Linux distribution. You can do this through your Linux package. On Ubuntu/Debian run ``sudo apt-get install git``. Once git is installed, `generate an SSH key and add it to GitHub <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux>`_. With git setup, the repository can be cloned into the directory of your choice with the following commands

.. code-block:: sh

   # Clone the repository
   git clone git@github.com:myerspat/pyMAISE.git

   # Checkout the develop branch
   git checkout develop

---------
Branching
---------

In the previous section we checked out the develop branch. This branch is the main branch of the repository and is never directly edited. Prior to writing your own code, create a new branch. Branches are always made off of develop, so prior to any new branch ensure to checkout develop again.

.. code-block:: sh

   # Checkout the develop branch
   git checkout develop

   # Get the latest version of the develop branch from GitHub
   git pull

   # Create your new working branch off of develop called `branch-name`
   git checkout -b branch-name

Pior to each branch, update your latest develop version with ``git pull``. Additionally, the ``branch-name`` can be anything you'd like and is preferably a name related to the changes/issue the branch is for. Now you can make edits to the repository code on your new branch. To keep your branch up to date with develop, run ``git pull origin develop``. As a best practice, a new branch should be made for each issue.

----------
Committing
----------

To see any changes you make to the source code reflected on the develop branch on GitHub, the code must be committed and pushed. Committing entails staging and then committing your staged changes with a short massage describing the changes you made.

.. code-block:: sh

   # Stage the changed file for committing
   git add path/to/file

   # Commit the changes with a short descriptive message
   git commit -m "what I changed"

Commit often and write strong messages so reviewers have an easier time understanding what was changed and why.

-------
Pushing
-------

Changes that have been committed can now be pushed assuming they pass all tests and the code runs without issues. To push your branch to GitHub run ``git push -u origin branch-name``, this will set an upstream link to the remote branch now on the server so futher changes can be pushed with just ``git push``. 

----------------
General Workflow
----------------

Changes should be made only if there is a representative issue in the issue tab of the GitHub repository with detailed information of what should change and why. The issue can then be assinged to a contributor, a branch can be made, and coding can begin. Once the branch is ready, it can be pushed to the remote repository and a pull request (PR) can be made for that branch to be pulled into develop. The PR should outline what changes were made and why as well as what issue the PR closes. The PR must then be reviewed by someone other than the original contributor. If the code passes all tests and the reviewer is happy with the work then the branch may be pulled into develop. The reviewer may request changes in which you should make the changes and push them. 

