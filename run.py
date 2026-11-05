
"""
One-command launcher for the Offline Document Intelligence Chatbot.

Run from the project root (the folder containing app/):

    python run.py

You will be asked to choose:
    1. Local access only
    2. Share with devices on the same Wi-Fi / LAN

Optional:
    python run.py --local-only
    python run.py --share-network
    python run.py --reload

Press Ctrl+C once to stop both services.

IMPORTANT:
- LAN sharing works only while this PC is running the application.
- Other devices must be able to reach this PC over the same network.
- Sharing a LAN URL does not install or execute the application
  on another person's computer.
- Internet-wide sharing requires a secure public deployment or tunnel.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = os.path.dirname(os.path.abspath(__file__))

# FastAPI stays private. Streamlit communicates with it locally.
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

UI_PORT = 8501
STARTUP_TIMEOUT = 120


def port_in_use(port, host="127.0.0.1"):
    """Check whether a TCP port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def get_lan_ip():
    """Find this computer's LAN IPv4 address without internet access."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]

    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "YOUR-PC-IP"

    finally:
        sock.close()


def choose_access_mode(args):
    """Choose whether Streamlit should be accessible over the LAN."""

    if args.local_only:
        return False

    if args.share_network:
        return True

    print("\n==========================================")
    print("  OFFLINE DOCUMENT INTELLIGENCE CHATBOT")
    print("==========================================")
    print("\nChoose application access mode:")
    print("1. Local access only (this PC)")
    print("2. Share with devices on the same Wi-Fi / LAN")

    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "1":
            return False

        if choice == "2":
            return True

        print("Invalid choice. Please enter 1 or 2.")


def wait_for_backend(proc, timeout=STARTUP_TIMEOUT):
    """Wait until FastAPI's health endpoint responds."""

    url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:

        if proc.poll() is not None:
            return None

        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )

        except Exception:
            time.sleep(0.5)

    return None


def report_health(health):
    """Report important local dependencies."""

    ollama = health.get("ollama") or {}

    if not ollama.get("available"):
        print(
            "\nWARNING: Ollama is not reachable."
            "\nStart Ollama before using chat and generation."
        )

    else:
        for kind in ("generation", "embedding"):
            model_name = ollama.get(f"{kind}_model")

            if model_name and not ollama.get(
                f"{kind}_model_available"
            ):
                print(
                    f"\nWARNING: {kind} model '{model_name}' "
                    f"is not installed."
                )
                print(f"Install it using: ollama pull {model_name}")

    if not (health.get("chromadb") or {}).get("available"):
        print("\nWARNING: ChromaDB is unavailable.")

    if not (health.get("metadata") or {}).get("available"):
        print("\nWARNING: Metadata database is unavailable.")


def stop_process(proc, name):
    """Stop a subprocess gracefully, then force-stop if needed."""

    if proc is None or proc.poll() is not None:
        return

    print(f"Stopping {name}...")

    proc.terminate()

    try:
        proc.wait(timeout=5)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():

    parser = argparse.ArgumentParser(
        description="Start FastAPI and Streamlit together."
    )

    access_group = parser.add_mutually_exclusive_group()

    access_group.add_argument(
        "--share-network",
        action="store_true",
        help="Allow devices on the same LAN to access Streamlit."
    )

    access_group.add_argument(
        "--local-only",
        action="store_true",
        help="Allow access only from this PC."
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart FastAPI automatically when code changes."
    )

    args = parser.parse_args()

    # Select access mode.
    share_network = choose_access_mode(args)

    # Streamlit is exposed only when LAN sharing is enabled.
    ui_host = "0.0.0.0" if share_network else "127.0.0.1"

    # Verify required project files.
    required_files = [
        "app/main.py",
        "app/ui/streamlit_app.py",
    ]

    for relative_path in required_files:

        full_path = os.path.join(ROOT, relative_path)

        if not os.path.isfile(full_path):
            print(f"\nERROR: Missing file: {relative_path}")
            print("Run run.py from the project root.")
            return 1

    # Check ports before starting either service.
    for port in (BACKEND_PORT, UI_PORT):

        if port_in_use(port):
            print(
                f"\nERROR: Port {port} is already in use."
                "\nClose the existing application and try again."
            )
            return 1

    if share_network:
        print("\nLAN sharing is ENABLED.")
        print("Only trusted devices should be allowed to connect.")
        print("There is no application-level authentication configured.")
    else:
        print("\nLocal-only access is ENABLED.")

    # FastAPI command.
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
    ]

    if args.reload:
        backend_cmd.append("--reload")

    # Streamlit command.
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/ui/streamlit_app.py",
        "--server.address",
        ui_host,
        "--server.port",
        str(UI_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    # Environment variables.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    # Streamlit's Python process communicates with FastAPI locally.
    ui_env = dict(env)
    ui_env["API_URL"] = (
        f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    )

    backend = None
    ui = None

    try:

        # Start FastAPI.
        print(
            f"\nStarting FastAPI on "
            f"http://{BACKEND_HOST}:{BACKEND_PORT} ..."
        )

        backend = subprocess.Popen(
            backend_cmd,
            cwd=ROOT,
            env=env,
        )

        # Wait for backend health.
        health = wait_for_backend(backend)

        if health is None:
            print("\nERROR: FastAPI failed to start.")
            print("Check the backend error messages.")
            return 1

        print("\nFastAPI started successfully.")

        report_health(health)

        # Start Streamlit.
        print(f"\nStarting Streamlit on port {UI_PORT} ...")

        ui = subprocess.Popen(
            ui_cmd,
            cwd=ROOT,
            env=ui_env,
        )

        time.sleep(2)

        if ui.poll() is not None:
            print("\nERROR: Streamlit failed to start.")
            return 1

        # Display access URLs.
        print("\n==========================================")
        print("  APPLICATION STARTED SUCCESSFULLY")
        print("==========================================")

        print(
            f"\nFastAPI (private): "
            f"http://{BACKEND_HOST}:{BACKEND_PORT}"
        )

        print(
            f"Streamlit (local): "
            f"http://localhost:{UI_PORT}"
        )

        if share_network:

            lan_ip = get_lan_ip()

            print(
                f"\nStreamlit (LAN): "
                f"http://{lan_ip}:{UI_PORT}"
            )

            print("\nShare the LAN URL with devices connected")
            print("to the same Wi-Fi / local network.")

            print("\nIMPORTANT:")
            print("- Keep this computer switched on.")
            print("- Keep this launcher running.")
            print("- Allow Streamlit through the Windows Firewall")
            print("  on private networks if Windows asks.")
            print("- All connected users access the same host data.")
            print("- No login or user isolation is configured.")

        else:
            print("\nNetwork sharing is disabled.")

        print("\nPress Ctrl+C to stop both services.")
        print("==========================================\n")

        # Monitor both services.
        while True:

            if backend.poll() is not None:
                print("\nFastAPI stopped unexpectedly.")
                return 1

            if ui.poll() is not None:
                print("\nStreamlit stopped unexpectedly.")
                return 1

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutdown requested.")

        return 0

    finally:
        stop_process(ui, "Streamlit")
        stop_process(backend, "FastAPI")

        print("\nApplication stopped.")


if __name__ == "__main__":
    sys.exit(main())
