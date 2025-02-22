import logging

# Define a custom log level for STATUS
STATUS_LEVEL = 30
logging.addLevelName(STATUS_LEVEL, "STATUS")

def _status(self, message, *args, **kwargs):
    if self.isEnabledFor(STATUS_LEVEL):
        self._log(STATUS_LEVEL, message, args, **kwargs)

# Add the status method to the logger
logging.Logger.status = _status

# Configure logging to look like a print statement with a STATUS indicator
def _setup_logging():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s", datefmt=None)