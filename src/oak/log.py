import os

if ("OAK_DEBUG" in os.environ) and (os.environ["OAK_DEBUG"] != "0"):
    from ._native.Debug.pyoaklog import *
else:
    from ._native.Release.pyoaklog import *
