import sys, subprocess

def package(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    import openai
except:
    package("openai")
    import openai

try:
    import dotenv
except:
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
    import art
except ImportError:
    package("art")
    import art

dotenv.load_dotenv()

__all__ = [
    "openai",
    "prompt_toolkit",
    "black",
    "art",
    "package"
]

