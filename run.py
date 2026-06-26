import subprocess, sys, os
from pathlib import Path

BASE_DIR = Path(__file__).parent

def install_requirements():
    req_file = BASE_DIR / "backend" / "requirements.txt"
    if not req_file.exists():
        print("requirements.txt not found!")
        return False
    print("Installing requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        capture_output=False
    )
    if result.returncode != 0:
        print("Failed to install requirements!")
        return False
    print("Requirements installed!\n")
    return True

def run_server():
    backend_dir = BASE_DIR / "backend"
    print("Starting ML Platform...")
    print("Open browser: http://localhost:8000\n")
    print("Press Ctrl+C to stop.\n")

    os.chdir(backend_dir)
    debug = os.getenv("DEBUG", "true").lower() == "true"
    args = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--port", os.getenv("PORT", "8000"),
        "--host", os.getenv("HOST", "0.0.0.0"),
    ]
    if debug:
        args.append("--reload")
    subprocess.run(args)

if __name__ == "__main__":
    print("=" * 45)
    print("        ML Platform — Starting Up")
    print("=" * 45 + "\n")

    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(str(env_file))
            print("Loaded .env configuration\n")
        except ImportError:
            pass

    if install_requirements():
        run_server()
    else:
        input("\nPress Enter to exit...")
