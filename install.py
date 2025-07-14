import sys, subprocess

def package(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    from openai import OpenAI
except:
    package("openai")
    from openai import OpenAI

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
    "OpenAI",
    "prompt_toolkit",
    "black",
    "art",
    "package"
]

