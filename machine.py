import os, sys, platform, datetime, getpass, time, subprocess

def package(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    from openai import OpenAI
except:
    package("openai")
    from openai import OpenAI

try:
    from dotenv import load_dotenv
except:
    package("python-dotenv")
    from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

tools = {
    "type": "function",
    "name": "python",
    "description": "Accepts a string of Python code as input, executes it securely, and returns its output as a string or code block. All side effects and state changes occur within the interpreter’s environment. Only valid, self-contained Python scripts are allowed.",
    "parameters": {
        "type": "object",
        "strict": True,
        "properties": {
            "input": {
                "type": "string",
                "description": "Python code to execute.",
            }
        },
        "required": ["input"],
    },
}

now = datetime.datetime.now()
osy = platform.system() + " " + platform.release()
arch = platform.machine()
host = platform.node()
user = getpass.getuser()
py = sys.executable
you = os.path.abspath(__file__)

def unique_key():
    return str(int(time.time()))

start = f"#<python_{unique_key()}>\n"
question = f"#<python_question_{unique_key()}>\n"
end = "#</python>"

def bld():
    return f"\n{unique_key()}\n{now}\n{osy}\n{arch}\n{host}\n{user}\n{py}\n{you}\n"

def aut(cmd):
    prompt = bld()
    prompt += start + open(os.path.abspath(__file__), encoding="utf-8").read() + end
    prompt += open("prompt.txt", encoding="utf-8").read()
    response = client.responses.create(
        instructions=prompt,
        model="gpt-4.1",
        tools=[tools],
        input=cmd
    )
    log(f"\nAgent: \n{response.output_text}\n")
    return response.output_text

def ext(x):
    if question in x:
        return x.partition(question)[2].partition(end)[0], True
    elif start in x:
        return x.partition(start)[2].partition(end)[0], False
    return x, False

def log(x):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(x + "\n")

if __name__ == "__main__":
    while 1:
        try:
            cmd = input("Agent: ")
            if cmd.lower() != "file":
                log(cmd)
            cmd = open("log.txt", encoding="utf-8").read().strip()
            cde = aut(cmd)
            rce, is_question = ext(cde)
            if is_question:
                pass
            exec(rce, globals())
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(str(e))
