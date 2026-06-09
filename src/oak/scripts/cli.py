import os
import sys
import subprocess
import signal

directory = "Debug" if "OAK_DEBUG" in os.environ else "Release"


def _run_binary(binary_name, prefix=f"../_bin/{directory}"):
    bin_path = os.path.join(os.path.dirname(__file__), prefix, binary_name)
    if os.name != "nt":
        signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    proc = subprocess.Popen([bin_path, *sys.argv[1:]])
    try:
        return_code = proc.wait()
    except KeyboardInterrupt:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        return_code = proc.wait()
    sys.exit(return_code)


def oak_search_test():
    _run_binary("search-test")


def vs():
    _run_binary("vs")


def generate():
    _run_binary("generate")


def benchmark():
    _run_binary("benchmark")


def chall():
    _run_binary("chall")


def evo():
    _run_binary("evo")
