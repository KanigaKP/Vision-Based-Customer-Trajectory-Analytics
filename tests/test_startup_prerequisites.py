"""
Unit tests for Startup Prerequisites & Server Port Management
=============================================================
Tests:
  1. Free port probe returns 'free'.
  2. Healthy Streamlit server probe returns 'healthy_server'.
  3. Occupied non-Streamlit / unresponsive port probe returns 'occupied_unknown'.
  4. Missing model specifically distinguishes Demonstration Ready vs. New Analysis Not Ready.
  5. Missing model does NOT claim all prerequisites passed.
"""

import sys
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

from scripts.check_prerequisites import (
    probe_port_status,
    check_python_version,
    check_dependencies,
    check_configs,
    check_saved_reports,
    check_models,
    check_media,
    main as checker_main,
)


def get_ephemeral_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class DummyHealthyStreamlitHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/_stcore/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logging


class DummyForeignHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Internal Server Error")

    def log_message(self, format, *args):
        pass


def test_probe_port_free():
    """Verify that an unbound port reports 'free'."""
    port = get_ephemeral_port()
    status = probe_port_status(port)
    assert status == "free"


def test_probe_port_healthy_server():
    """Verify that a server responding with 200 on /_stcore/health reports 'healthy_server'."""
    port = get_ephemeral_port()
    server = HTTPServer(("127.0.0.1", port), DummyHealthyStreamlitHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        status = probe_port_status(port)
        assert status == "healthy_server"
    finally:
        server.shutdown()
        server.server_close()


def test_probe_port_occupied_unknown():
    """Verify that a non-Streamlit / erroring server on port reports 'occupied_unknown'."""
    port = get_ephemeral_port()
    server = HTTPServer(("127.0.0.1", port), DummyForeignHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        status = probe_port_status(port)
        assert status == "occupied_unknown"
    finally:
        server.shutdown()
        server.server_close()


def test_prerequisite_launcher_exit_codes():
    """Test launcher mode exit codes for free, healthy, and occupied ports."""
    port_free = get_ephemeral_port()
    with patch("sys.argv", ["check_prerequisites.py", "--port", str(port_free), "--launcher"]):
        code = checker_main()
        assert code == 0  # Ready for fresh launch

    # Healthy server -> code 2
    port_healthy = get_ephemeral_port()
    server_h = HTTPServer(("127.0.0.1", port_healthy), DummyHealthyStreamlitHandler)
    th_h = threading.Thread(target=server_h.serve_forever, daemon=True)
    th_h.start()
    try:
        with patch("sys.argv", ["check_prerequisites.py", "--port", str(port_healthy), "--launcher"]):
            code = checker_main()
            assert code == 2  # Reuse healthy server
    finally:
        server_h.shutdown()
        server_h.server_close()

    # Foreign occupied port -> code 3
    port_occ = get_ephemeral_port()
    server_o = HTTPServer(("127.0.0.1", port_occ), DummyForeignHandler)
    th_o = threading.Thread(target=server_o.serve_forever, daemon=True)
    th_o.start()
    try:
        with patch("sys.argv", ["check_prerequisites.py", "--port", str(port_occ), "--launcher"]):
            code = checker_main()
            assert code == 3  # Abort due to conflict
    finally:
        server_o.shutdown()
        server_o.server_close()


def test_missing_model_distinguishes_demo_from_new_analysis(capsys):
    """Verify that when yolo11n.onnx is missing, it does not claim all prerequisites passed."""
    port_free = get_ephemeral_port()
    with patch("scripts.check_prerequisites.check_models", return_value=(False, None)):
        with patch("sys.argv", ["check_prerequisites.py", "--port", str(port_free)]):
            code = checker_main()
            captured = capsys.readouterr().out
            assert "[PASS] Saved-Results Demonstration: READY" in captured
            assert "[WARN] New-Analysis Pipeline: NOT READY" in captured
            assert "All system prerequisites verified (Both Demonstration & New Analysis ready)" not in captured
            assert "[PARTIAL SUCCESS] Saved-Results Demonstration is fully functional." in captured
            assert code == 0
