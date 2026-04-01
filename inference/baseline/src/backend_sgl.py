import subprocess
import time
import os
import signal
from .engine import BaseEngine

class SGLangEngine(BaseEngine):
    def __init__(self, model_path: str, tp_size: int, extra_args: list = None):
        super().__init__(model_path, tp_size)
        self.extra_args = extra_args or []
        self.process = None

    def start_server(self):
        print(f"🚀 Starting SGLang server for {self.model_path} with TP={self.tp_size}")
        
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--tp", str(self.tp_size)
        ] + self.extra_args

        # We redirect stdout to devnull to avoid cluttering the terminal,
        # but stderr is captured to check for immediate failures.
        self.process = subprocess.Popen(
            cmd,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.PIPE
        )
        
        # Naive wait - in a prod env, you'd poll an HTTP endpoint like /health
        print("⏳ Waiting for SGLang server to initialize (approx 30s)...")
        time.sleep(30)
        
        # Check if process crashed immediately
        if self.process.poll() is not None:
            raise RuntimeError(f"SGLang Server crashed on startup. Exit code: {self.process.returncode}")

        print("✅ SGLang server seems up!")

    def stop_server(self):
        if self.process and self.process.poll() is None:
            print("🛑 Stopping SGLang server...")
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
            else:
                os.kill(self.process.pid, signal.SIGKILL)
            self.process.wait()
            self.process = None
            print("🛑 SGLang server stopped.")
        else:
            print("⚠️ SGLang server was already stopped or not started.")
