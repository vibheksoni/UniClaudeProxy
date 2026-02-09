import threading
import logging
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger("anyclaude")


class _ConfigReloadHandler(FileSystemEventHandler):
    """Handles config file modification events and triggers reload callback.

    Attributes:
        config_path: Path - Resolved path to the config file.
        config_name: str - Filename of the config file.
        callback: callable - Function to invoke on config change.
    """

    def __init__(self, config_path: str, callback):
        """Initialize the handler.

        Args:
            config_path: str - Path to the config file to watch.
            callback: callable - Reload callback function.
        """
        self.config_path = Path(config_path).resolve()
        self.config_name = self.config_path.name
        self.callback = callback
        self._lock = threading.Lock()

    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification events, triggering reload for config changes.

        Args:
            event: FileModifiedEvent - The filesystem event.
        """
        if event.is_directory:
            return

        event_path = Path(event.src_path).resolve()
        if event_path.name == self.config_name:
            with self._lock:
                logger.info("Config file changed: %s", self.config_name)
                try:
                    self.callback()
                    logger.info("Config reloaded successfully")
                except Exception as e:
                    logger.error("Failed to reload config: %s", e)


class ConfigWatcher:
    """Watches config.json for changes and triggers hot reload.

    Attributes:
        config_path: Path - Resolved path to the config file.
        callback: callable - Function to invoke on config change.
        observer: Observer | None - The watchdog observer instance.
    """

    def __init__(self, config_path: str, callback):
        """Initialize the watcher.

        Args:
            config_path: str - Path to the config file to watch.
            callback: callable - Function to call when config changes.
        """
        self.config_path = Path(config_path).resolve()
        self.callback = callback
        self.observer = None

    def start(self):
        """Start watching the config file for modifications."""
        handler = _ConfigReloadHandler(str(self.config_path), self.callback)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.config_path.parent), recursive=False)
        self.observer.start()
        logger.info("Watching config: %s", self.config_path)

    def stop(self):
        """Stop watching the config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Config watcher stopped")
