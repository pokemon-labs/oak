import os

if ("OAK_DEBUG" in os.environ) and (os.environ["OAK_DEBUG"] != "0"):
    from ._native.Debug.pyoaksearch import *
else:
    from ._native.Release.pyoaksearch import *
