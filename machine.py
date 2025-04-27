# This is you. You are this program:

import os, sys, platform, datetime, socket, getpass, re, subprocess
def package(p):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', p])
try:
    from openai import OpenAI
except:
    package('openai')
    from openai import OpenAI
try:
    from dotenv import load_dotenv
except:
    package('python-dotenv')
    from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
tpc = {
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
ip = socket.gethostbyname(socket.gethostname())
user = getpass.getuser()
py = sys.executable
you = os.path.abspath(__file__)
def bld():
    return f"\n{now}\n{osy}\n{arch}\n{host}\n{ip}\n{user}\n{py}\n{you}\n"
def aut(cmd):
    ins = open("prompt.txt", encoding="utf-8").read()
    ins += "\n```python\n" + open("machine.py", encoding="utf-8").read() + "\n```"
    cmdf = bld()
    msg = f"Write Python code that fully accomplishes the command:\n{cmd}\n{cmdf}"
    response = client.responses.create(
        instructions=ins,
        model = "gpt-4.1",
        tools = [tpc],
        input = msg
    )
    log(f"\n{response.output_text}")
    return response.output_text
def ext(x):
    m = re.search(r"```python\n(.*?)```", x, re.S)
    if m:
        return m.group(1)
    return x
def log(x):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(x + "\n")
if __name__ == "__main__":
    while 1:
        try:
            cmd = input("Machine: ")
            if cmd.lower() == "file":
                cmd = open("file.txt", encoding="utf-8").read()
                
            cde = aut(cmd)
            rce = ext(cde)
            exec(rce,globals())
        except KeyboardInterrupt:
            break 
        except Exception as e:
            log(str(e))
