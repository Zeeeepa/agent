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

def bld():
    global unique_key, code, question, reflect, end
    unique_key = str(int(time.time()))
    question = f"<python_question_{unique_key}>\n"
    reflect = f"<python_reflect_{unique_key}>\n"
    code = f"<python_{unique_key}>\n"
    end = "</python>"
    return f"\n{unique_key}\n{now}\n{osy}\n{arch}\n{host}\n{user}\n{py}\n{you}\n"

def aut(cmd):
    prompt = bld()
    prompt += code + open(os.path.abspath(__file__), encoding="utf-8").read() + end
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
        return x.partition(question)[2].partition(end)[0], 1
    elif reflect in x:
        return x.partition(reflect)[2].partition(end)[0], 2
    elif code in x:
        return x.partition(code)[2].partition(end)[0], 3
    else:
        return x, 0

def log(x):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(x + "\n")

def process(cmd):
    while 1:
        try:
            cde = aut(cmd)
            rce, state = ext(cde)
            if state == 1:
                pass
            elif state == 2:
                cmd = open("log.txt", encoding="utf-8").read().strip()
                continue
            else:
                pass
            exec(rce, globals())
            break
        except Exception as e:
            log(str(e))

if __name__ == "__main__":
    while 1:
        try:
            cmd = input("Agent: ")
            log(cmd)
            cmd = open("log.txt", encoding="utf-8").read().strip()
            process(cmd)
        except KeyboardInterrupt:
            break
