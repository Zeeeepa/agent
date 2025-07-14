import os, sys, subprocess, platform, getpass, time, traceback, asyncio, re, io, ast, multiprocessing, contextlib, threading

from install import openai, prompt_toolkit, black, art, package

proxy = os.getenv("PROXY")
if proxy:
    try:
        from httpx_socks import SyncProxyTransport
        import httpx
    except ImportError:
        package("httpx-socks")
        from httpx_socks import SyncProxyTransport
        import httpx
    client = openai.OpenAI(http_client=httpx.Client(transport=SyncProxyTransport.from_url(proxy)))
else:
    client = openai.OpenAI()

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

def fix_and_format_code(code):
    try:
        tree = ast.parse(code)
        code = ast.unparse(tree)
    except Exception:
        pass
    try:
        return black.format_str(code, mode=black.Mode())
    except Exception:
        return code

dangerous = [
    "while True:",
    "while 1:",
    "for _ in iter(int, 1):",
    "time.sleep(",
    "threading.Thread(",
    "subprocess.run(",
    "subprocess.call(",
]

def truncate_output(output, limit=100_000):
    if len(output) <= limit:
        return output
    notice = f"\n...[truncated {len(output) - limit} chars]...\n"
    remaining = limit - len(notice)
    if remaining <= 0:
        return notice
    half = remaining // 2
    return output[:half] + notice + output[-half:]

def worker(code, g, ret):
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            exec(code, g)
            ret['error'] = None
        except Exception:
            ret['error'] = traceback.format_exc()
    ret['output'] = truncate_output(f.getvalue())

def execute_safely(code):
    code = fix_and_format_code(code)
    log(code, "code.txt")
    def run_in_process():
        with multiprocessing.Manager() as mgr:
            ret = mgr.dict()
            p = multiprocessing.Process(target=worker, args=(code, {}, ret))
            p.start()
            p.join()
            output = ret.get('output', '')
            error = ret.get('error')
            if output:
                log(output, "terminal.txt")
            if error:
                log(error)
    if any(danger in code for danger in dangerous):
        threading.Thread(target=run_in_process, daemon=True).start()
    else:
        exec(code, globals())

async def execute_safely_async(code):
    await asyncio.to_thread(execute_safely, code)

async def process(cmd):
    global pulse
    while True:
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
                await execute_safely_async(code)
                pulse += 10
            break
        except Exception:
            cmd = open("memory.txt", encoding="utf-8").read().strip()
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
        prompt_task = asyncio.create_task(s.prompt_async("", key_bindings=k))
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
    art.tprint("Jinx", "random")
    while True:
        try:
            cmd = await inp()
            if cmd is None:
                cmd = "<no_response>"
            if not cmd.strip():
                continue
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