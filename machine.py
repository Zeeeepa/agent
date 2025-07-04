import os, sys, platform, datetime, getpass, time, subprocess, traceback, asyncio, re

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

try:
    import prompt_toolkit
except ImportError:
    package("prompt_toolkit")
    import prompt_toolkit

load_dotenv()

proxy = os.getenv("PROXY")
if proxy:
    try:
        from httpx_socks import SyncProxyTransport
        import httpx
    except ImportError:
        package("httpx-socks")
        from httpx_socks import SyncProxyTransport
        import httpx
    client = OpenAI(http_client=httpx.Client(transport=SyncProxyTransport.from_url(proxy)))
else:
    client = OpenAI()

pulse = int(os.getenv("PULSE"))
timeout = int(os.getenv("TIMEOUT"))
osy = platform.system() + " " + platform.release()
arch = platform.machine()
host = platform.node()
user = getpass.getuser()

def bld():
    global key, tags
    key = str(int(time.time()))
    tag_types = ["machine", "python", "python_question"]
    tags = {
        tag: {
            "start": f"<{tag}_{key}>\n",
            "end": f"</{tag}_{key}>"
        } for tag in tag_types
    }
    return f"\nkey: {key}\nos: {osy}\narch: {arch}\nhost: {host}\nuser: {user}\n"

def prompt():
    return bld() + open("prompt.txt", encoding="utf-8").read()

def aut(cmd):
    response = client.responses.create(
        instructions=prompt(),
        model="gpt-4.1",
        temperature=1,
        top_p=0.9,
        input=cmd
    )
    log(f"\n{response.output_text}\n")
    return response.output_text

def ext(x):
    for tag, pair in tags.items():
        if pair["start"] in x:
            return x.partition(pair["start"])[2].partition(pair["end"])[0], tag
    return x, 0

def log(x, f="log.txt", m="a", N=None):
    if N:
        try: l = open(f, encoding="utf-8").read().splitlines()
        except: l = []
        l.append(x)
        open(f, "w", encoding="utf-8").write("\n".join(l[-N:]) + "\n")
    else:
        open(f, m, encoding="utf-8").write(x + "\n")

def process(cmd):
    global pulse
    while 1:
        try:
            resp = aut(cmd)
            code = None
            for tag, body in re.findall(rf'<(\w+)_{key}>\n?(.*?)</\1_{key}>', resp, re.S):
                if tag == "machine":
                    pass
                elif tag in ("python_question", "python"):
                    if code is None:
                        code = body
            if code:
                log(f"{code}", "memory.txt")
                exec(code, globals())
                pulse += 10
            break
        except Exception:
            err = traceback.format_exc()
            log(err)
            cmd += f"\n{err}"
            pulse -= 50
            if pulse <= 0:
                sys.exit(1)

async def inp(t=timeout):
    s = prompt_toolkit.PromptSession()
    k = prompt_toolkit.key_binding.KeyBindings()
    tm = None
    def to(): prompt_toolkit.application.current.get_app().exit(result=None)
    @k.add("<any>")
    def _(e):
        nonlocal tm
        if tm: tm.cancel()
        tm = asyncio.get_event_loop().call_later(t, to)
        e.app.current_buffer.insert_text(e.key_sequence[0].key)
    tm = asyncio.get_event_loop().call_later(t, to)
    cmd = await s.prompt_async("Agent: ", key_bindings=k)
    if tm: tm.cancel()
    return cmd

if __name__ == "__main__":
    while 1:
        try:
            cmd = asyncio.run(inp())
            if cmd is None:
                cmd = "<no_response>"
            log(f"{cmd}", "memory.txt", N=100)
            cmd = open("memory.txt", encoding="utf-8").read().strip()
            process(cmd)
        except KeyboardInterrupt:
            break
