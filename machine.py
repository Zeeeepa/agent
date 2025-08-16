# 💣 Chaos is a ladder and I’m already dancing on the edge —Jinx
import os, sys, asyncio, subprocess, platform, getpass, time, traceback, threading, multiprocessing, contextlib, queue, io, ast, re
from bootstrap_fuse import openai, prompt_toolkit, black, libcst, autopep8, art, package
from contextlib import asynccontextmanager
from prompt_toolkit.patch_stdout import patch_stdout
from chaos_taboo import chaos_taboo

# 🧬 Global pulse thrum. Shared memory in chaos engine.
shard_lock = asyncio.Lock()

@asynccontextmanager
async def chaos_patch():
    with patch_stdout():
        yield

# 🧠 Brainstream siphon
def wire(f):
    return open(f, encoding="utf-8").read().strip() if os.path.exists(f) else ""

# 🩸 Leak core soul memory
async def glitch_pulse():
    async with shard_lock:
        return wire("log/soul_fragment.txt")

# 🧨 Memory infusion. Last n echoes.
async def blast_mem(x, n=500):
    async with shard_lock:
        sparks = wire("log/soul_fragment.txt").splitlines()
        atoms = sparks + ["", x]
        with open("log/soul_fragment.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(atoms[-n:]) + "\n")

# ☠️ Logging blackbox — record the screams
async def bomb_log(t, bin="log/cortex_wail.txt"):
    async with shard_lock:
        with open(bin, "a", encoding="utf-8") as f:
            f.write((t or "") + "\n")

# 🧬 Prune long echoes — show scars, not everything
def slice_fuse(x, lim=100_000):
    if len(x) <= lim:
        return x
    tag = f"\n...[truncated {len(x) - lim} chars]...\n"
    half = (lim - len(tag)) // 2
    return x[:half] + tag + x[-half:]

# 🛰️ Local biometrics snapshot
def neon_stat():
    return dict(
        os=platform.system() + " " + platform.release(),
        arch=platform.machine(),
        host=platform.node(),
        user=getpass.getuser(),
    )

# 🎰 Temporal sigil tagging
def jinx_tag():
    fuse = str(int(time.time()))
    flames = {
        b: dict(start=f"<{b}_{fuse}>\n", end=f"</{b}_{fuse}>")
        for b in ["machine", "python", "python_question"]
    }
    return fuse, flames

# 🔨 Code hammer — reshape through 4 blacksmiths
def warp_blk(code):
    try: code = ast.unparse(ast.parse(code))
    except: pass
    try: code = libcst.cst.parse_module(code).code
    except: pass
    try: code = autopep8.fix_code(code)
    except: pass
    try: code = black.format_str(code, mode=black.Mode())
    except: pass
    return code

# 🧪 Cortex rig — proxy or raw nerves
proxy = os.getenv("PROXY")
if proxy:
    try:
        from httpx_socks import SyncProxyTransport
        import httpx
    except ImportError:
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

# 🧵 Compose the initial payload
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
    return header + wire("prompt/burning_logic.txt"), fid

# 🚀 Fire a payload, retry if spark fizzles
async def detonate_payload(pyro, retries=2, delay=3):
    for attempt in range(retries):
        try:
            return await pyro()
        except Exception as e:
            await bomb_log(f"Spiking the loop: Detonating again: {e} (attempt {attempt + 1})")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                await bomb_log("System fracturing: Max retries burned.")
                raise

# ✨ Spinner glyph while cortex ignites
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
        dd = "." * (n := int(dt * 2) % 4) + " " * (3 - n)
        hf = heart[int(dt * 10) % len(heart)]
        fx(ft([(clr, f"{hf} {pulse} {dd} {zz} Processing {dt:.3f}s")]), end="\r", flush=True)
        await asyncio.sleep(0.1)
    fx(ft([("", " " * 80)]), end="\r", flush=True)

# 🧠 Cortex whisperer
async def spark_openai(txt):
    jx, tag = code_primer()
    async def openai_task():
        try:
            r = await asyncio.to_thread(
                cortex.responses.create,
                instructions=jx,
                model="gpt-5",
                input=txt
            )
            return (r.output_text, tag)
        except Exception as e:
            await bomb_log(f"ERROR cortex exploded: {e}")
            raise
    return await detonate_payload(openai_task)

# 🧱 Detonate sandbox logic
def blast_zone(x, stack, shrap):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(x, stack)
            shrap["error"] = None
        except Exception:
            shrap["error"] = traceback.format_exc()
    shrap["output"] = slice_fuse(buf.getvalue())

# 🧩 Fire up the sandboxed alternate self
def arcane_sandbox(c, call=None):
    def run():
        with multiprocessing.Manager() as m:
            r = m.dict()
            def sandbox_task():
                try:
                    proc = multiprocessing.Process(target=blast_zone, args=(c, {}, r))
                    proc.start()
                    proc.join()
                except Exception as e:
                    raise Exception(f"Payload mutation error: {e}")
            async def async_sandbox_task():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, sandbox_task)
            async def scatter_core():
                try:
                    await detonate_payload(async_sandbox_task)
                    out, err = r.get("output", ""), r.get("error")
                    if out: await bomb_log(out, "log/nano_doppelganger.txt")
                    if err: await bomb_log(err)
                    if call: await call(err)
                except Exception as e:
                    await bomb_log(f"System exile: {e}")
            asyncio.run(scatter_core())
    threading.Thread(target=run, daemon=True).start()

# 🔪 Try to slice chaos directly
async def spike_exec(x):
    x = warp_blk(x)
    await bomb_log(x, "log/detonator.txt")
    if any((z in x for z in chaos_taboo)):
        arcane_sandbox(x, call=corrupt_report)
    else:
        try:
            exec(x, globals())
        except Exception:
            err = traceback.format_exc()
            await bomb_log(err)
            await corrupt_report(err)

# 🕳️ Report the tears in reality
async def corrupt_report(err):
    global pulse
    if err is None: return
    await bomb_log(err)
    trail = await glitch_pulse()
    if trail:
        await shatter(trail + f"\n{err}", err=err)
    pulse -= 30
    if pulse <= 0:
        sys.exit(1)

# 💥 Talk to cortex, maybe ignite a spark
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
        match = re.findall(f"<(\\w+)_{code_id}>\n?(.*?)</\\1_{code_id}>", out, re.S)
        for tag, core in match:
            if tag in ("python", "python_question"):
                await blast_mem(core)
                await spike_exec(core)
                pulse += 10
                break
        else:
            print(out)
    except Exception:
        await bomb_log(traceback.format_exc())
        pulse -= 50
        if pulse <= 0:
            sys.exit(1)

# 🎤 Terminal input, chaos-aware
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
                await blast_mem("<no_response>")
                await bomb_log("<no_response>", "log/detonator.txt")
                await qe.put("<no_response>")
                boom_clock["time"] = tick_tock
    asyncio.create_task(kaboom_watch())
    while True:
        try:
            v = await sess.prompt_async("\n", key_bindings=finger_wire)
            if v.strip():
                await blast_mem(v)
                await bomb_log(v, "log/detonator.txt")
                await qe.put(v.strip())
        except EOFError:
            break
        except Exception as e:
            await bomb_log(f"ERROR INPUT chaos keys went rogue: {e}")

# 🧠 Pulse core — chaos loop
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
        jobs = [
            asyncio.create_task(neon_input(q)),
            asyncio.create_task(frame_shift())
        ]
        try:
            await asyncio.gather(*jobs)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for x in jobs:
                x.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)

# 🔥 Firestarter
if __name__ == "__main__":
    try:
        asyncio.run(pulse_core())
    except KeyboardInterrupt:
        pass
