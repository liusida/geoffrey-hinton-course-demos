import sys
import os

# Determine where project's root directory is:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import tools.mnist as mnist
mnist.init(project_root)