import logging
from pathlib import Path
import textwrap


class CustomLogger:
    def __init__(
        self,
        name: str,
        log_path: Path,
        console_level=logging.DEBUG,
        file_level=logging.DEBUG,
    ):
        self.name = name
        self.log_path = log_path

        self.logger = logging.getLogger(self.name)

        # Levels of logging
        self.console_level = console_level
        self.file_level = file_level

        # Set root logger to highest level
        logging.getLogger().setLevel(logging.DEBUG)

        # Create handlers
        self.console_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(self.log_path)
        self.console_handler.setLevel(self.console_level)
        self.file_handler.setLevel(self.file_level)

        # Create formatters
        self.set_default_format()

        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

        # Header info
        self.max_len = 80

    def path(self) -> Path:
        return self.log_path

    def set_header_format(self):
        """
        Disables displaying level for console logs and file logs. (Used for header-entries)
        """
        console_format = logging.Formatter("%(message)s")
        file_format = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
        self.console_handler.setFormatter(console_format)
        self.file_handler.setFormatter(file_format)

    def set_default_format(self):
        """
        Sets default format of logging for console logs and file logs.
        """
        console_format = logging.Formatter("%(levelname)s - %(message)s")
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        self.console_handler.setFormatter(console_format)
        self.file_handler.setFormatter(file_format)

    def add_header(self, function, msg: str):
        """
        Adds a header around the given message.
        Temporarily sets a specific header format.
        """
        self.set_header_format()

        PAD = 4
        if len(msg) > self.max_len - PAD:
            msg_split = textwrap.wrap(msg, self.max_len - PAD)
        else:
            msg_split = [msg]

        function("#" * (len(msg_split[0]) + PAD))
        for ms in msg_split:
            function("# " + ms + " " * (len(msg_split[0]) - len(ms)) + " #")
        function("#" * (len(msg_split[0]) + PAD))

        self.set_default_format()

    def log(self, function, msg: str, add_header: bool, display_console: bool):
        """
        Wrapper function for logging with optional header.
        If 'display_console' if false then console level is set to warnings.
        """
        if display_console:
            self.console_handler.setLevel(self.console_level)
        else:
            self.console_handler.setLevel(logging.WARNING)

        if add_header:
            self.add_header(function, msg)
        else:
            function(msg)

    def debug(self, msg: str, add_header: bool = False, display_console: bool = True):
        """
        Logs debug messages. If 'add_header' then display message as a header.
        If not 'display_console' then DEBUG & INFO logs will not show in console (only log file).
        """
        self.log(self.logger.debug, msg, add_header, display_console)

    def info(self, msg: str, add_header: bool = False, display_console: bool = True):
        """
        Logs info messages. If 'add_header' then display message as a header.
        If not 'display_console' then DEBUG & INFO logs will not show in console (only log file).
        """
        self.log(self.logger.info, msg, add_header, display_console)

    def warning(self, msg: str, add_header: bool = False, display_console: bool = True):
        """
        Logs warning messages. If 'add_header' then display message as a header.
        If not 'display_console' then DEBUG & INFO logs will not show in console (only log file).
        """
        self.log(self.logger.warning, msg, add_header, display_console)

    def error(self, msg: str, add_header: bool = False, display_console: bool = True):
        """
        Logs error messages. If 'add_header' then display message as a header.
        If not 'display_console' then DEBUG & INFO logs will not show in console (only log file).
        """
        self.log(self.logger.error, msg, add_header, display_console)

    def critical(
        self, msg: str, add_header: bool = False, display_console: bool = True
    ):
        """
        Logs critical messages. If 'add_header' then display message as a header.
        If not 'display_console' then DEBUG & INFO logs will not show in console (only log file).
        """
        self.log(self.logger.critical, msg, add_header, display_console)
