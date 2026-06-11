import os

if ("OAK_DEBUG" in os.environ) and (os.environ["OAK_DEBUG"] != "0"):
    from ._native.Debug.pyoaktrain import *
else:
    from ._native.Release.pyoaktrain import *
