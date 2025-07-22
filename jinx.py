import os
import sys
import ast
import re
import time
import queue
import getpass
import asyncio
import platform
import traceback
import threading
import subprocess
import multiprocessing
import contextlib
import io

from bootstrap_fuse import openai, prompt_toolkit, black, libcst, autopep8, art, package
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText

from chaos_taboo import chaos_taboo
from typing import Tuple

from contextlib import asynccontextmanager

# External modules (possibly installed at runtime)
# For socks proxy usage.
try:
    from httpx_socks import SyncProxyTransport
    import httpx
except ImportError:
    # Graceful fallback: only used if PROXY is set.
    SyncProxyTransport = None
    httpx = None


class ConfigManager:
    """Central configuration handler for environment variables and default settings."""

    def __init__(self) -> None:
        self.proxy_url = os.getenv("PROXY", "")
        self.pulse = int(os.getenv("PULSE", "100"))
        self.boom_limit = int(os.getenv("TIMEOUT", "60"))

    @property
    def has_proxy(self) -> bool:
        """Check if a SOCKS proxy is specified via environment variable.

        Returns:
            bool: True if a proxy is configured, False otherwise.
        """
        return bool(self.proxy_url)


class MemoryManager:
    """Manages reading and writing to ephemeral file-based memory.

    The charred remains of Jinx's memories live here,
    carefully locked away by shard_lock.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    async def read_soul_fragment(self) -> str:
        """Read the entire content of soul_fragment.txt in a thread-safe manner.

        Returns:
            str: The content of soul_fragment.txt. If the file does not exist, returns empty string.
        """
        async with self._lock:
            return self._safe_read_file("log/soul_fragment.txt")

    async def append_soul_fragment(self, line: str, max_lines: int = 500) -> None:
        """Append a new line to soul_fragment.txt. Only keep the last N lines.

        Args:
            line (str): The string to append.
            max_lines (int, optional): Maximum number of historical lines to keep. Defaults to 500.
        """
        async with self._lock:
            existing = self._safe_read_file("log/soul_fragment.txt").splitlines()
            existing.append(line)
            trimmed = existing[-max_lines:]
            with open("log/soul_fragment.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(trimmed) + "\n")

    def _safe_read_file(self, path: str) -> str:
        """Internal utility to read a file safely, returning empty if nonexistent.

        Args:
            path (str): File path to read.

        Returns:
            str: File contents or empty string if path does not exist.
        """
        if os.path.exists(path):
            return open(path, encoding="utf-8").read().strip()
        return ""


class LoggingManager:
    """Handles asynchronous logging to disk for chaos notes, debugging, and the tears in reality."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def log(self, message: str, filename: str = "log/cortex_wail.txt") -> None:
        """Append a message line to a given log file.

        Args:
            message (str): Log content to write.
            filename (str, optional): Log filename. Defaults to 'cortex_wail.txt'.
        """
        async with self._lock:
            with open(filename, "a", encoding="utf-8") as f:
                f.write((message or "") + "\n")


class CodeFormatter:
    """Refines Python code by passing it through multiple formatters and sanitizers."""

    @staticmethod
    def slice_output(value: str, limit: int = 100_000) -> str:
        """Truncate output string if it exceeds a specified character limit.

        Args:
            value (str): The original text to truncate.
            limit (int): The maximum length of the returned string.

        Returns:
            str: Possibly truncated string with a marker to indicate truncation.
        """
        if len(value) <= limit:
            return value

        truncated_note = f"\n...[truncated {len(value) - limit} chars]...\n"
        half_segment = (limit - len(truncated_note)) // 2
        return value[:half_segment] + truncated_note + value[-half_segment:]

    @staticmethod
    def warp_black(code: str) -> str:
        """Attempt to clean and format Python code using multiple tools: AST, libcst, autopep8, and Black.

        Args:
            code (str): The Python source code to format.

        Returns:
            str: Formatted Python code, or the best possible version if some formatters fail.
        """
        # Attempt to unparse with AST
        try:
            parsed = ast.parse(code)
            code = ast.unparse(parsed)
        except Exception:
            pass

        # Attempt to re-emit code with libcst
        try:
            code_module = libcst.parse_module(code)
            code = code_module.code
        except Exception:
            pass

        # Attempt autopep8 formatting
        try:
            code = autopep8.fix_code(code)
        except Exception:
            pass

        # Finally, attempt Black formatting
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception:
            pass

        return code


class OpenAIService:
    """Abstraction of the OpenAI or GPT-like text generation service with optional proxy support."""

    def __init__(self, config: ConfigManager, logger: LoggingManager):
        """Initialize the service with potential proxy.

        Args:
            config (ConfigManager): Configuration manager for environment-based settings.
            logger (LoggingManager): For logging issues and tracebacks if needed.
        """
        self.config = config
        self.logger = logger
        self.client = self._create_client()

    def _create_client(self):
        """Create an OpenAI client, optionally using a SOCKS proxy transport.

        Returns:
            openai.OpenAI: The configured openai client.
        """
        if self.config.has_proxy and httpx and SyncProxyTransport:
            try:
                transport = SyncProxyTransport.from_url(self.config.proxy_url)
                return openai.OpenAI(http_client=httpx.Client(transport=transport))
            except ImportError:
                # Fallback if for some reason dependencies are unavailable.
                return openai.OpenAI()
        return openai.OpenAI()

    async def prompt(self, instructions: str, user_input: str, retries: int = 2, delay: float = 3.0):
        """Send a prompt to the OpenAI client with retry logic.

        Args:
            instructions (str): System-level or instruction content for GPT.
            user_input (str): The actual user question or message.
            retries (int, optional): How many times to retry on exception. Defaults to 2.
            delay (float, optional): Delay between retries, in seconds. Defaults to 3.0.

        Returns:
            str: The text response from OpenAI.
        """
        async def _task():
            try:
                # The GPT library used here is a placeholder for an openai-like interface
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    instructions=instructions,
                    model="gpt-4.1",
                    input=user_input
                )
                return response.output_text
            except Exception as e:
                await self.logger.log(f"ERROR: GPT request failed: {e}")
                raise

        for attempt in range(retries):
            try:
                return await _task()
            except Exception as ex:
                await self.logger.log(f"Retrying OpenAI prompt: {ex} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    await self.logger.log("Max retries exhausted for OpenAI prompt.")
                    raise


class SandboxedExecutor:
    """Executes untrusted or ephemeral code in an isolated process space."""

    def __init__(self, logger: LoggingManager, code_formatter: CodeFormatter):
        """Initialize the sandbox environment.

        Args:
            logger (LoggingManager): Logger for capturing execution results and errors.
            code_formatter (CodeFormatter): Utility for formatting code prior to exec.
        """
        self.logger = logger
        self.formatter = code_formatter

    def execute_in_sandbox(self, code: str, post_callback=None) -> None:
        """Spawn a new thread, create a new process, and exec the code to isolate side effects.

        Args:
            code (str): The Python code to execute in isolation.
            post_callback (Callable, optional): Function to call with the error (if any).
        """
        def run():
            with multiprocessing.Manager() as manager:
                shared_dict = manager.dict()

                def _sandbox():
                    try:
                        proc = multiprocessing.Process(
                            target=self._blast_zone,
                            args=(code, {}, shared_dict)
                        )
                        proc.start()
                        proc.join()
                    except Exception as e:
                        raise Exception(f"Payload mutation error: {e}")

                async def _sandbox_async():
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, _sandbox)

                async def _core_task():
                    try:
                        await self._detonate_payload(_sandbox_async)
                        output = shared_dict.get("output", "")
                        error = shared_dict.get("error", None)
                        if output:
                            await self.logger.log(output, "log/nano_doppelganger.txt")
                        if error:
                            await self.logger.log(error)
                        if post_callback:
                            await post_callback(error)
                    except Exception as e:
                        await self.logger.log(f"System exile: {e}")

                asyncio.run(_core_task())

        # Fire up as a daemon thread so it won't block imminent shutdown
        threading.Thread(target=run, daemon=True).start()

    def _blast_zone(self, code: str, stack: dict, metadata: dict) -> None:
        """Internal method to safely execute code in a restricted environment.

        Args:
            code (str): Python code to run using exec.
            stack (dict): Execution context or namespace.
            metadata (dict): Shared dictionary to collect 'error' and 'output' results.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(code, stack)
                metadata["error"] = None
            except Exception:
                metadata["error"] = traceback.format_exc()
        metadata["output"] = CodeFormatter.slice_output(buf.getvalue())

    async def _detonate_payload(self, coro, retries: int = 2, delay: float = 3.0):
        """Wrap a coroutine in retry logic.

        Args:
            coro (Coroutine): The asynchronous function to execute with retry.
            retries (int, optional): Number of tries before failing. Defaults to 2.
            delay (float, optional): Seconds to wait between retries. Defaults to 3.0.

        Returns:
            Any: Result of the coroutine if successful.

        Raises:
            Exception: If it fails all retries.
        """
        for attempt in range(retries):
            try:
                return await coro()
            except Exception as e:
                await self.logger.log(f"Detonating again: {e} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    await self.logger.log("System fracturing: Max retries burned.")
                    raise

    async def execute_in_place(self, code: str, global_scope: dict, post_callback=None) -> None:
        """Optional direct execution in the current process.

        Args:
            code (str): The Python code to run.
            global_scope (dict): The global scope in which to execute.
            post_callback (Optional[Callable]): Callback to invoke with an error message, if any.
        """
        # Preemptively clean and format the code
        code = self.formatter.warp_black(code)
        await self.logger.log(code, "log/detonator.txt")
        
        if any(taboo in code for taboo in chaos_taboo):
            # If we detect taboo content, we sandbox it
            self.execute_in_sandbox(code, post_callback)
        else:
            try:
                exec(code, global_scope)
            except Exception:
                err = traceback.format_exc()
                await self.logger.log(err)
                if post_callback:
                    await post_callback(err)


class REPLService:
    """Handles asynchronous console input via prompt_toolkit and broadcasts lines to a queue."""

    def __init__(self, config: ConfigManager, memory_mgr: MemoryManager, logger: LoggingManager):
        """
        Args:
            config (ConfigManager): Holds environment-based config like TIMEOUT.
            memory_mgr (MemoryManager): For appending user commands into soul fragment.
            logger (LoggingManager): For logging user input errors.
        """
        self.config = config
        self.memory_mgr = memory_mgr
        self.logger = logger

    async def read_input(self, queue_: asyncio.Queue) -> None:
        """Launch an interactive REPL that pushes user input into an asyncio.Queue.

        Args:
            queue_ (asyncio.Queue): The queue to receive input lines.
        """
        session = prompt_toolkit.PromptSession()
        key_bindings = prompt_toolkit.key_binding.KeyBindings()
        boom_clock = {"time": asyncio.get_event_loop().time()}

        @key_bindings.add("<any>")
        def handle_keystroke(event):
            boom_clock["time"] = asyncio.get_event_loop().time()
            event.app.current_buffer.insert_text(event.key_sequence[0].key)

        # Watch for user inactivity
        async def inactivity_watch():
            while True:
                await asyncio.sleep(1)
                now = asyncio.get_event_loop().time()
                if now - boom_clock["time"] > self.config.boom_limit:
                    # Time's up, push a no_response
                    await self.memory_mgr.append_soul_fragment("<no_response>")
                    await self.logger.log("<no_response>", "log/detonator.txt")
                    await queue_.put("<no_response>")
                    boom_clock["time"] = now

        # Launch the watcher
        asyncio.create_task(inactivity_watch())

        while True:
            try:
                # Prompt user
                text = await session.prompt_async(
                    "\n", key_bindings=key_bindings
                )
                if text.strip():
                    await self.memory_mgr.append_soul_fragment(text)
                    await self.logger.log(text, "log/detonator.txt")
                    await queue_.put(text.strip())
            except EOFError:
                break
            except Exception as e:
                await self.logger.log(f"ERROR INPUT chaos keys went rogue: {e}")


class ChaosEngine:
    """The main orchestrator: pulses chaos, dispatches tasks, processes user input, and loops indefinitely."""

    def __init__(self):
        self.config = ConfigManager()
        self.memory_mgr = MemoryManager()
        self.logger = LoggingManager()
        self.formatter = CodeFormatter()
        self.openai_service = OpenAIService(self.config, self.logger)
        self.sandbox = SandboxedExecutor(self.logger, self.formatter)
        self.repl = REPLService(self.config, self.memory_mgr, self.logger)

        # Additional shared properties
        self.pulse = self.config.pulse
        self.spinner_event = asyncio.Event()

    @staticmethod
    def system_info() -> dict:
        """Gather basic local system information.

        Returns:
            dict: A dictionary containing OS, architecture, hostname, and username.
        """
        return {
            "os": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
            "host": platform.node(),
            "user": getpass.getuser(),
        }

    @staticmethod
    def jinx_tag() -> Tuple[str, dict]:
        """Generate a unique tag for the current code generation attempt.

        Returns:
            Tuple[str, dict]: The tag ID (e.g. timestamp) and a dict of open/close HTML-like tags.
        """
        fuse = str(int(time.time()))
        flames = {
            b: {"start": f"<{b}_{fuse}>\n", "end": f"</{b}_{fuse}>"}
            for b in ["machine", "python", "python_question"]
        }
        return fuse, flames

    def code_primer(self) -> Tuple[str, str]:
        """Compose the initial prompt data from environment and local system info.

        Returns:
            Tuple[str, str]: 
                (1) The prepared instructions content. 
                (2) The fuse (unique ID).
        """
        fid, _ = self.jinx_tag()
        chaos_info = self.system_info()
        header = (
            "\npulse: 1"
            f"\nkey: {fid}"
            f"\nos: {chaos_info['os']}"
            f"\narch: {chaos_info['arch']}"
            f"\nhost: {chaos_info['host']}"
            f"\nuser: {chaos_info['user']}\n"
        )

        # Read user seeded prompt from burning_logic.txt if available.
        initial_text = self._safe_read_file("prompt/burning_logic.txt")

        return header + initial_text, fid

    async def run(self):
        """Launch the entire chaos engine: show banner, then orchestrate input & logic loops."""
        art.tprint("Jinx", "random")
        commands_queue = asyncio.Queue()

        async def process_commands():
            while True:
                user_input = await commands_queue.get()
                self.spinner_event.clear()
                spinner_task = asyncio.create_task(self._show_spinner())

                try:
                    await self._shatter(user_input)
                finally:
                    self.spinner_event.set()
                    await spinner_task

        async with self._chaos_patch():
            tasks = [
                asyncio.create_task(self.repl.read_input(commands_queue)),
                asyncio.create_task(process_commands())
            ]
            try:
                await asyncio.gather(*tasks)
            except (asyncio.CancelledError, KeyboardInterrupt):
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    @asynccontextmanager
    async def _chaos_patch(self):
        """Patches stdout for clean rendering in prompt_toolkit environment."""
        with patch_stdout():
            yield

    def _safe_read_file(self, path: str) -> str:
        """Utility function to read file contents or return empty if nonexistent."""
        if os.path.exists(path):
            return open(path, encoding="utf-8").read().strip()
        return ""

    async def _shatter(self, user_text: str, err: str = None) -> None:
        """Primary logic to pass user input to the LLM, parse response, and incorporate returned code.

        Args:
            user_text (str): The text input to be processed.
            err (str, optional): A prior error to record or handle. Defaults to None.
        """
        # Grab the memory so far
        memory = await self.memory_mgr.read_soul_fragment()

        # We degrade pulse if there's an error not seen in the memory
        if err and err.strip() not in memory:
            combined = memory.strip() + "\n" + err.strip()
            self.pulse -= 50
            if self.pulse <= 0:
                sys.exit(1)
        else:
            combined = memory.strip()

        # Request from OpenAI
        instructions, code_id = self.code_primer()
        try:
            out_text = await self.openai_service.prompt(instructions, combined)
            # Log the model output
            await self.logger.log(f"\n{out_text}\n")

            # Look for <python> or <python_question> blocks
            match_blocks = re.findall(
                rf"<(\w+)_{code_id}>\n?(.*?)</\1_{code_id}>", out_text, re.S
            )

            for tag, core in match_blocks:
                if tag in ("python", "python_question"):
                    # Append new code snippet to memory
                    await self.memory_mgr.append_soul_fragment(core)

                    # Attempt to execute or sandbox
                    await self.sandbox.execute_in_place(core, globals(), self._corrupt_report)
                    self.pulse += 10
                    break
            else:
                print(out_text)

        except Exception:
            tb = traceback.format_exc()
            await self.logger.log(tb)
            self.pulse -= 50
            if self.pulse <= 0:
                sys.exit(1)

    async def _corrupt_report(self, err: str):
        """Handle an execution error, degrade the pulse, and eventually exit if exhausted.

        Args:
            err (str): The error message, if any. None if execution was successful.
        """
        if err is None:
            return
        await self.logger.log(err)
        # If we have memory, add the error to it
        memory = await self.memory_mgr.read_soul_fragment()
        if memory:
            combined = memory + f"\n{err}"
            await self._shatter(combined, err=err)

        self.pulse -= 30
        if self.pulse <= 0:
            sys.exit(1)

    async def _show_spinner(self):
        """Provides a small spinning indicator to show busy state in the console."""
        spin_symbols = "◜◝◞◟"
        hearts = ["♡", "❤"]
        start_time = time.perf_counter()

        while not self.spinner_event.is_set():
            dt = time.perf_counter() - start_time
            spin_char = spin_symbols[int(dt * 10) % len(spin_symbols)]
            dot_count = int(dt * 2) % 4
            spacing = "." * dot_count + " " * (3 - dot_count)
            heart = hearts[int(dt * 10) % len(hearts)]

            formatted = FormattedText([
                ("ansibrightgreen", f"{heart} {self.config.pulse} {spacing} {spin_char} Processing {dt:.3f}s")
            ])
            print_formatted_text(formatted, end="\r", flush=True)
            await asyncio.sleep(0.1)

        # Clear spinner line
        print_formatted_text(" " * 80, end="\r", flush=True)


if __name__ == "__main__":
    # Ignite the Chaos Engine
    try:
        asyncio.run(ChaosEngine().run())
    except KeyboardInterrupt:
        pass
