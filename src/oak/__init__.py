import os

if ("OAK_DEBUG" in os.environ) and (os.environ["OAK_DEBUG"] != "0"):
    print("OAK DEBUG")
    from ._native.Debug.pyoak import *
else:
    from ._native.Release.pyoak import *

from . import util
from . import common_args

__version__ = "1.1.2"
