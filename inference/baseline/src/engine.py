from abc import ABC, abstractmethod

class BaseEngine(ABC):
    def __init__(self, model_path: str, tp_size: int):
        self.model_path = model_path
        self.tp_size = tp_size
        self.process = None

    @abstractmethod
    def start_server(self):
        """Start the inference backend process via subprocess."""
        pass

    @abstractmethod
    def stop_server(self):
        """Terminate the server process."""
        pass

    def restart_server(self):
        """Helper to cleanly stop and start the server when recovering from OOMs."""
        self.stop_server()
        self.start_server()

