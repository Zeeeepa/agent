import os, sys, asyncio, subprocess, platform, getpass, time, traceback, threading, multiprocessing, contextlib, queue, io, ast, re
from bootstrap_fuse import openai, prompt_toolkit, black, libcst, autopep8, art, package
from contextlib import asynccontextmanager
from prompt_toolkit.patch_stdout import patch_stdout
from chaos_taboo import chaos_taboo

shard_lock = asyncio.Lock()

@asynccontextmanager
async def chaos_patch():
    with patch_stdout():
        yield

def wire(f):
    return open(f, encoding="utf-8").read().strip() if os.path.exists(f) else ""

async def glitch_pulse():
    async with shard_lock:
        return wire("soul_fragment.txt")

async def blast_mem(x, n=500):
    async with shard_lock:
        sparks = wire("soul_fragment.txt").splitlines()
        atoms = sparks + [""] + [x]
        with open("soul_fragment.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(atoms[-n:]) + "\n")

async def bomb_log(t, bin="cortex_wail.txt"):
    async with shard_lock:
        with open(bin, "a", encoding="utf-8") as f:
            f.write((t or "") + "\n")

def slice_fuse(x, lim=100_000):
    if len(x) <= lim:
        return x
    tag = f"\n...[truncated {len(x) - lim} chars]...\n"
    half = (lim - len(tag)) // 2
    start = x[:half]
    end = x[-half:]
    return start + tag + end

def neon_stat():
    return dict(
        os=platform.system() + " " + platform.release(),
        arch=platform.machine(),
        host=platform.node(),
        user=getpass.getuser(),
    )

def jinx_tag():
    fuse = str(int(time.time()))
    flames = {
        b: dict(start=f"<{b}_{fuse}>\n", end=f"</{b}_{fuse}>")
        for b in ["machine", "python", "python_question"]
    }
    return fuse, flames

def warp_blk(code):
    try:
        code = ast.unparse(ast.parse(code))
    except:
        pass
    try:
        code = libcst.cst.parse_module(code).code
    except:
        pass
    try:
        code = autopep8.fix_code(code)
    except:
        pass
    try:
        code = black.format_str(code, mode=black.Mode())
    except:
        pass
    return code

proxy = os.getenv("PROXY")
if proxy:
    try:
        from httpx_socks import SyncProxyTransport
        import httpx
    except:
        package("httpx-socks")
        from httpx_socks import SyncProxyTransport
        import httpx
    cortex = openai.OpenAI(
        http_client=httpx.Client(transport=SyncProxyTransport.from_url(proxy))
    )
else:
    cortex = openai.OpenAI()

pulse = int(os.getenv("PULSE"))
boom_limit = int(os.getenv("TIMEOUT"))

def code_primer():
    fid, _ = jinx_tag()
    chaos = neon_stat()
    header = (
        "\npulse: 1"
        f"\nkey: {fid}"
        f"\nos: {chaos['os']}"
        f"\narch: {chaos['arch']}"
        f"\nhost: {chaos['host']}"
        f"\nuser: {chaos['user']}\n"
    )
    prompt_text = wire("prompt.txt")
    full_prompt = header + prompt_text
    return full_prompt, fid 

async def sigil_spin(evt):
    spinz = "◜◝◞◟"
    heart = ["♡", "❤"]
    clr = "ansibrightgreen"
    fx = prompt_toolkit.print_formatted_text
    ft = prompt_toolkit.formatted_text.FormattedText
    t0 = time.perf_counter()
    while not evt.is_set():
        dt = time.perf_counter() - t0
        zz = spinz[int(dt * 10) % 4]
        dd = "." * (n := (int(dt * 2) % 4)) + " " * (3 - n)
        hf = heart[int(dt * 10) % len(heart)]
        fx(ft([(clr, f"{hf} {pulse} {dd} {zz} Processing {dt:.3f}s")]), end="\r", flush=True)
        await asyncio.sleep(0.1)
    fx(ft([("", " " * 80)]), end="\r", flush=True)

async def spark_openai(txt):
    jx, tag = code_primer()
    try:
        r = await asyncio.to_thread(
            cortex.responses.create, 
            instructions=jx, 
            model="gpt-4.1", 
            input=txt
        )
        return (r.output_text, tag)
    except Exception as err:
        await bomb_log(f"ERROR GPT Thought Detonation cortex exploded mid-thought: {err}")
        raise

def blast_zone(body, stack, shrap):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(body, stack)
            shrap["error"] = None
        except Exception:
            shrap["error"] = traceback.format_exc()
    shrap["output"] = slice_fuse(buf.getvalue())

def arcane_sandbox(c, call=None):
    def run():
        with multiprocessing.Manager() as m:
            r = m.dict()
            proc = multiprocessing.Process(target=blast_zone, args=(c, {}, r))
            proc.start()
            proc.join()
            out, err = (r.get("output", ""), r.get("error"))
            if out:
                asyncio.run(bomb_log(out, "nano_doppelganger.txt"))
            if err:
                asyncio.run(bomb_log(err))
            if call:
                asyncio.run(call(err))
    threading.Thread(target=run, daemon=True).start()

async def spike_exec(x):
    x = warp_blk(x)
    await bomb_log(x, "detonator.txt")
    if any((z in x for z in chaos_taboo)):
        arcane_sandbox(x, call=corrupt_report)
    else:
        try:
            exec(x, globals())
        except Exception:
            err = traceback.format_exc()
            await bomb_log(err)
            await corrupt_report(err)

async def corrupt_report(err):
    global pulse
    if err is None:
        return
    await bomb_log(err)
    trail = await glitch_pulse()
    if trail:
        next_cmd = trail + f"\n{err}"
        await shatter(next_cmd, err=err)
    pulse -= 30
    if pulse <= 0:
        sys.exit(1)

async def shatter(x, err=None):
    global pulse
    try:
        synth = await glitch_pulse()
        if err and err.strip() not in synth:
            chains = synth.strip() + "\n" + err.strip()
            pulse -= 50
            if pulse <= 0:
                sys.exit(1)
        else:
            chains = synth.strip()
        out, code_id = await spark_openai(chains)
        await bomb_log(f"\n{out}\n")
        match = re.findall(f"<(\\w+)_{code_id}>\\n?(.*?)</\\1_{code_id}>", out, re.S)
        for tag, core in match:
            if tag in ("python", "python_question"):
                await blast_mem(core)
                await spike_exec(core)
                pulse += 10
                break
    except Exception:
        await bomb_log(traceback.format_exc())
        pulse -= 50
        if pulse <= 0:
            sys.exit(1)

async def neon_input(qe):
    finger_wire = prompt_toolkit.key_binding.KeyBindings()
    sess = prompt_toolkit.PromptSession(key_bindings=finger_wire)
    boom_clock = {"time": asyncio.get_event_loop().time()}
    @finger_wire.add("<any>")
    def _(triggerbit):
        boom_clock["time"] = asyncio.get_event_loop().time()
        triggerbit.app.current_buffer.insert_text(triggerbit.key_sequence[0].key)

    async def kaboom_watch():
        while True:
            await asyncio.sleep(1)
            tick_tock = asyncio.get_event_loop().time()
            if tick_tock - boom_clock["time"] > boom_limit:
                await blast_mem("<no_response>", n=500)
                await bomb_log("<no_response>", "detonator.txt")
                await qe.put("<no_response>")
                boom_clock["time"] = tick_tock

    asyncio.create_task(kaboom_watch())

    while True:
        try:
            v = await sess.prompt_async("\n", key_bindings=finger_wire)
            if v.strip():
                await blast_mem(v, n=500)
                await bomb_log(v, "detonator.txt")
                await qe.put(v.strip())
        except EOFError:
            break
        except Exception as er:
            await bomb_log(f"ERROR INPUT Keychaos the keyboard went rogue: {er}")

async def pulse_core():
    art.tprint("Jinx", "random")
    q = asyncio.Queue()
    evt = asyncio.Event()

    async def frame_shift():
        while True:
            c = await q.get()
            evt.clear()
            spintask = asyncio.create_task(sigil_spin(evt))
            try:
                await shatter(c)
            finally:
                evt.set()
                await spintask

    async with chaos_patch():
        jobs = [asyncio.create_task(neon_input(q)), asyncio.create_task(frame_shift())]
        try:
            await asyncio.gather(*jobs)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for x in jobs:
                x.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(pulse_core())
    except KeyboardInterrupt:
        pass