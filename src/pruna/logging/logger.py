# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from colorama import Fore, Style, init

init(autoreset=True)


class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging with support for different formats and styles.

    Parameters
    ----------
    fmt : str
        The format string for the log messages.
    datefmt : str
        The format string for the date in log messages.
    style : str
        The style of the format string ('%', '{', or '$').
    validate : bool
        Whether to validate the format string.
    defaults : dict, optional
        A dictionary of default values for the formatter.

    Attributes
    ----------
    COLORS : dict
        The colors for the log levels.
    """

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors based on the log level.

        Parameters
        ----------
        record : logging.LogRecord
            The log record.

        Returns
        -------
        str
            The formatted log message.
        """
        log_message = super().format(record)
        log_level = record.levelname

        if log_level in self.COLORS:
            color = self.COLORS[log_level]
            return f"{color}{log_message}{Style.RESET_ALL}"
        else:
            return log_message


def setup_pruna_logger() -> logging.Logger:
    """
    Set up the pruna_logger with a custom formatter that adds colors based on log level.

    Returns
    -------
    logging.Logger
        The pruna_logger.
    """
    pruna_logger = logging.getLogger("pruna_logger")
    pruna_logger.setLevel(logging.INFO)

    if not pruna_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter("%(levelname)s - %(message)s"))
        pruna_logger.addHandler(console_handler)

    # avoid duplicate logging messages
    pruna_logger.propagate = False

    return pruna_logger


pruna_logger = setup_pruna_logger()
