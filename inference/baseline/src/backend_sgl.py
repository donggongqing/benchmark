import subprocess
import time
import os
import signal
from .engine import BaseEngine

class SGLangEngine(BaseEngine):
    def __init__(self, model_path: str, tp_size: int, extra_args: list = None, log_dir=None):
        super().__init__(model_path, tp_size)
        self.extra_args = extra_args or []
        self.process = None
        self.server_cmd = ""
        self.log_dir = log_dir
        self._log_file = None

    def start_server(self):
        print(f"🚀 Starting SGLang server for {self.model_path} with TP={self.tp_size}")
        
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--tp", str(self.tp_size)
        ] + self.extra_args

        self.server_cmd = " ".join(cmd)

        # Redirect server output to log file to keep terminal clean
        popen_kwargs = {}
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, "sglang_server.log")
            self._log_file = open(log_path, "a")
            popen_kwargs["stdout"] = self._log_file
            popen_kwargs["stderr"] = self._log_file
            print(f"📄 Server logs redirected to: {log_path}")

        self.process = subprocess.Popen(cmd, **popen_kwargs)
        
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
            self.server_cmd = ""
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            print("🛑 SGLang server stopped.")
        else:
            print("⚠️ SGLang server was already stopped or not started.")
