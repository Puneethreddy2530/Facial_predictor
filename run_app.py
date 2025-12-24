import subprocess
import os
import sys
import time
import webbrowser
import signal

def run():
    print("🚀 Launching Facial Prediction App...")

    # Set environment variables
    os.environ['MOCK_PREDICTION'] = '0'
    os.environ['PYTHONPATH'] = '.'
    
    # Determine python executable path
    # Check for local venv first (Prioritize .venv311)
    if sys.platform == 'win32':
        venv311_python = os.path.join(os.getcwd(), '.venv311', 'Scripts', 'python.exe')
        venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe')
        
        if os.path.exists(venv311_python):
            venv_python = venv311_python
    else:
        # Linux/Mac logic
        venv311_python = os.path.join(os.getcwd(), '.venv311', 'bin', 'python')
        venv_python = os.path.join(os.getcwd(), '.venv', 'bin', 'python')
        
        if os.path.exists(venv311_python):
            venv_python = venv311_python
        
    if os.path.exists(venv_python):
        python_exe = venv_python
        print(f"✅ Found Virtual Environment: {python_exe}")
    else:
        python_exe = sys.executable
        print(f"⚠️  Virtual Environment not found, using system python: {python_exe}")

    
    try:
        # Start Backend
        print("Starting Backend Service (Port 8001)...")
        # Using list format for subprocess avoids shell injection and quoting issues
        backend = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8001"],
            cwd=os.getcwd()
        )
        
        # Start Frontend
        print("Starting Frontend Server (Port 3000)...")
        frontend = subprocess.Popen(
            [python_exe, "-m", "http.server", "3000", "--directory", "frontend"],
            cwd=os.getcwd()
        )
        
        # Wait a moment for services to initialize
        print("Waiting for services to start...")
        time.sleep(2)
        
        # Check if they are still running
        if backend.poll() is not None:
             print("❌ Backend failed to start.")
             frontend.terminate()
             sys.exit(1)

        # Open Web Browser
        print("🌐 Opening Browser...")
        webbrowser.open("http://localhost:3000")
        
        print("\n✅ App is Running!")
        print("Press Ctrl+C to stop all services.\n")
        
        # Keep main process alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup child processes
        if 'backend' in locals() and backend.poll() is None:
            backend.terminate()
            print("Backend stopped.")
        if 'frontend' in locals() and frontend.poll() is None:
            frontend.terminate()
            print("Frontend stopped.")
        print("Goodbye!")

if __name__ == "__main__":
    run()
