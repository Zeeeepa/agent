import os, sys, platform, getpass, time, subprocess, traceback, asyncio, re, io, tokenize

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

try:
    import black
except ImportError:
    package("black")
    import black

try:
    from art import tprint
except ImportError:
    package("art")
    from art import tprint

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

tag_types = [
    "machine",
    "python",
    "python_question",
]
def build():
    global key, tags
    key = str(int(time.time()))
    tags = {
        tag: {
            "start": f"<{tag}_{key}>\n",
            "end": f"</{tag}_{key}>"
        } for tag in tag_types
    }
    return key, tags

async def show_loading(stop_event: asyncio.Event):
    spin = ["◜", "◝", "◞", "◟"]
    start = time.perf_counter()
    while not stop_event.is_set():
        elapsed = time.perf_counter() - start
        sys.stdout.write(f"\r{spin[int(elapsed*8)%4]}  Processing {elapsed:.1f}s")
        sys.stdout.flush()
        await asyncio.sleep(0.05)
    sys.stdout.write("\r\033[K")

def prompt():
    build()
    meta = (
        f"\npulse: 1"
        f"\nkey: {key}"
        f"\nos: {osy}"
        f"\narch: {arch}"
        f"\nhost: {host}"
        f"\nuser: {user}\n"
    )
    return meta + open("prompt.txt", encoding="utf-8").read()

async def aut(cmd):
    stop_event = asyncio.Event()
    loading_task = asyncio.create_task(show_loading(stop_event))
    try:
        response = await asyncio.to_thread(
            client.responses.create,
            instructions=prompt(),
            model="gpt-4.1",
            temperature=1,
            top_p=0.9,
            input=cmd
        )
        log(f"\n{response.output_text}\n")
        return response.output_text
    finally:
        stop_event.set()
        await loading_task

def log(x, f="log.txt", m="a", N=None):
    if N:
        try: l = open(f, encoding="utf-8").read().splitlines()
        except: l = []
        l.append(x)
        open(f, "w", encoding="utf-8").write("\n".join(l[-N:]) + "\n")
    else:
        open(f, m, encoding="utf-8").write(x + "\n")

def split_semicolons_safe(code):
    result = []
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    for toknum, tokval, *_ in tokens:
        if toknum == tokenize.OP and tokval == ';':
            result.append((tokenize.NL, '\n'))
        else:
            result.append((toknum, tokval))
    return tokenize.untokenize(result)

dangerous = [
    "while True:",
    "while 1:",
    "for _ in iter(int, 1):",
    "time.sleep(",
    "threading.Thread(",
    "subprocess.run(",
    "subprocess.call(",
]
mode = black.Mode()
def execute_safely(code):
    code = split_semicolons_safe(code)
    code = black.format_str(code, mode=mode)
    log(f"{code}", "code.txt")
    is_dangerous = any(pattern in code for pattern in dangerous)
    if is_dangerous:
        with open("buffer.py", "w") as f:
            f.write(code)
        with open("error.txt", "w") as err:
            proc = subprocess.Popen(
                ["python", "buffer.py"],
                stdout=subprocess.DEVNULL,
                stderr=err
            )
            time.sleep(0.2)
            retcode = proc.poll()
            if retcode is not None and retcode != 0:
                with open("error.txt", "r") as err:
                    error_proc = err.read()
                raise RuntimeError(error_proc)
    else:
        exec(code, globals())

async def process(cmd):
    global pulse
    while 1:
        try:
            resp = await aut(cmd)
            code = None
            for tag, body in re.findall(rf'<(\w+)_{key}>\n?(.*?)</\1_{key}>', resp, re.S):
                if tag == "machine":
                    pass
                elif tag in ("python_question", "python"):
                    if code is None:
                        code = body
            if code:
                log(f"{code}", "memory.txt")
                execute_safely(code)
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
    timer_task = None
    done = asyncio.Event()
    async def timeout_worker():
        try:
            await asyncio.sleep(t)
            done.set()
        except asyncio.CancelledError:
            pass
    @k.add("<any>")
    def _(event):
        nonlocal timer_task
        if timer_task:
            timer_task.cancel()
        timer_task = asyncio.create_task(timeout_worker())
        event.app.current_buffer.insert_text(event.key_sequence[0].key)
    async def cancel_wait(task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    try:
        timer_task = asyncio.create_task(timeout_worker())
        prompt_task = asyncio.create_task(s.prompt_async("Agent: ", key_bindings=k))
        done_wait_task = asyncio.create_task(done.wait())
        done, _ = await asyncio.wait([prompt_task, done_wait_task], return_when=asyncio.FIRST_COMPLETED)
        if done_wait_task in done:
            await cancel_wait(prompt_task)
            await cancel_wait(timer_task)
            await cancel_wait(done_wait_task)
            return None
        else:
            await cancel_wait(timer_task)
            await cancel_wait(done_wait_task)
            return await prompt_task
    except asyncio.CancelledError:
        await cancel_wait(prompt_task)
        await cancel_wait(timer_task)
        await cancel_wait(done_wait_task)
        raise

async def main():
    tprint("Jinx", "random")
    while 1:
        try:
            cmd = await inp()
            if cmd is None:
                cmd = "<no_response>"
            log(f"{cmd}", "memory.txt", N=100)
            log(f"{cmd}", "code.txt")
            cmd = open("memory.txt", encoding="utf-8").read().strip()
            await process(cmd)
        except KeyboardInterrupt:
            break
        except asyncio.CancelledError:
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass