"""OSC bridge: Max/MSP <-> Python module routing.

Listens on a UDP port for OSC messages and dispatches to registered handlers.
Each handler is keyed by an address pattern (e.g. /mvp_a/perturb).

Dependencies: python-osc
    pip install python-osc
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient
except ImportError as e:  # pragma: no cover
    raise ImportError("python-osc not installed. Run: pip install python-osc") from e


logger = logging.getLogger(__name__)

Handler = Callable[..., None]


@dataclass
class OSCBridgeConfig:
    listen_host: str = "127.0.0.1"
    listen_port: int = 7400
    send_host: str = "127.0.0.1"
    send_port: int = 7401


@dataclass
class OSCBridge:
    config: OSCBridgeConfig = field(default_factory=OSCBridgeConfig)
    _dispatcher: Dispatcher = field(default_factory=Dispatcher, init=False)
    _server: ThreadingOSCUDPServer | None = field(default=None, init=False)
    _client: SimpleUDPClient | None = field(default=None, init=False)

    def register(self, address: str, handler: Handler) -> None:
        """Register an OSC address pattern to a handler callable."""
        self._dispatcher.map(address, handler)
        logger.info("OSC handler registered: %s", address)

    def send(self, address: str, *args: Any) -> None:
        """Send an OSC message back to Max/MSP."""
        if self._client is None:
            self._client = SimpleUDPClient(self.config.send_host, self.config.send_port)
        self._client.send_message(address, list(args))

    def serve_forever(self) -> None:
        """Block and serve OSC traffic until interrupted."""
        self._server = ThreadingOSCUDPServer(
            (self.config.listen_host, self.config.listen_port), self._dispatcher
        )
        logger.info(
            "OSC server listening on %s:%d",
            self.config.listen_host,
            self.config.listen_port,
        )
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
