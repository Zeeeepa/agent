import sys, subprocess

def package(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    import openai
except ImportError:
    package("openai")
    import openai

try:
    import dotenv
except ImportError:
    package("python-dotenv")
    import dotenv

try:
    import prompt_toolkit
except ImportError:
    package("prompt_toolkit")
    import prompt_toolkit

try:
    import black
except ImportError:
    package("black")
    import black

try:
    import libcst
except ImportError:
    package("libcst")
    import libcst

try:
    import autopep8
except ImportError:
    package("autopep8")
    import autopep8

try:
    import art
except ImportError:
    package("art")
    import art

dotenv.load_dotenv()

__all__ = [
    "openai",
    "prompt_toolkit",
    "black",
    "libcst",
    "autopep8",
    "art",
    "package"
]

