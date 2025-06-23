import os, sys, platform, datetime, getpass, time, subprocess, traceback

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

pulse = 100
now = datetime.datetime.now()
osy = platform.system() + " " + platform.release()
arch = platform.machine()
host = platform.node()
user = getpass.getuser()
py = sys.executable
you = os.path.abspath(__file__)

def bld():
    global unique_key, tags
    unique_key = str(int(time.time()))
    tag_types = ["python", "python_question", "python_reflect"]
    tags = {
        tag: {
            "start": f"<{tag}_{unique_key}>\n",
            "end": f"</{tag}_{unique_key}>"
        } for tag in tag_types
    }
    return f"\n{unique_key}\n{now}\n{osy}\n{arch}\n{host}\n{user}\n{py}\n{you}\n"

def prompt():
    return bld() + tags["python"]["start"] + open(os.path.abspath(__file__), encoding="utf-8").read() + tags["python"]["end"] + open("prompt.txt", encoding="utf-8").read()

def aut(cmd):
    response = client.responses.create(
        instructions=prompt(),
        model="gpt-4.1",
        tools=[tools],
        input=cmd
    )
    log(f"\nPulse: {pulse} \n{response.output_text}\n")
    return response.output_text

def ext(x):
    for tag, pair in tags.items():
        if pair["start"] in x:
            return x.partition(pair["start"])[2].partition(pair["end"])[0], tag
    return x, 0

def log(x):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(x + "\n")

def process(cmd):
    global pulse
    while 1:
        try:
            cde = aut(cmd)
            rce, state = ext(cde)
            if state == "python_question":
                pass
            elif state == "python_reflect":
                cmd = open("log.txt", encoding="utf-8").read().strip()
                continue
            else:
                pass
            exec(rce, globals())
            pulse += 10
            break
        except Exception:
            log(traceback.format_exc())
            pulse -= 50
            if pulse <= 0:
                sys.exit(1)

if __name__ == "__main__":
    while 1:
        try:
            cmd = input("Agent: ")
            log(cmd)
            cmd = open("log.txt", encoding="utf-8").read().strip()
            process(cmd)
        except KeyboardInterrupt:
            break
