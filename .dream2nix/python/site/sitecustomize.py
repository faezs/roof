import sys
import site

try:
  import _sitecustomize
except ImportError:
  pass

site.addsitedir("/Users/faezs/roof/.dream2nix/python/site")

# addsitedir only supports appending to the path, not prepending.
# As we already include a non-editable instance of each package
# in our pyEnv, those would shadow the editables. So we move
# the editables to the front of sys.path.
for index, path in enumerate(sys.path):
  if path in ['/Users/faezs/roof']:
    sys.path.insert(0, sys.path.pop(index))
        