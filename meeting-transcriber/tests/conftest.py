import signal

import pytest


@pytest.fixture(autouse=True)
def _signal_handlers_as_they_were():
    """follow() installs SIGINT/SIGTERM/SIGHUP handlers and never puts the old ones back (its process ends after it): in
    pytest they would outlive the test, so a later test's SIGINT would no longer raise KeyboardInterrupt (it would record
    to its limit instead) and pytest could no longer be stopped by SIGTERM or Ctrl+C. Each test gets them back."""
    names = [name for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)]
    saved = {name: signal.getsignal(getattr(signal, name)) for name in names}
    yield
    for name, handler in saved.items():
        if handler is not None:                      # None: a handler not installed from Python, which cannot be put back
            signal.signal(getattr(signal, name), handler)
