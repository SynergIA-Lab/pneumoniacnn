import os
import sys
import subprocess
import platform

def main():
    print("==================================================")
    
    # 1. Create virtual environment
    if not os.path.exists(".venv"):
        print("Creating virtual environment (.venv)...")
        try:
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("Virtual environment created.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    else:
        print(".venv folder already exists. Skipping creation.")

    # 2. Determine paths based on OS
    is_windows = platform.system() == "Windows"
    if is_windows:
        pip_path = os.path.join(".venv", "Scripts", "pip.exe")
        activate_cmd = ".venv\\Scripts\\Activate.ps1"
    else:
        pip_path = os.path.join(".venv", "bin", "pip")
        activate_cmd = "source .venv/bin/activate"

    # Verify pip exists in virtual environment
    if not os.path.exists(pip_path):
        print(f"Error: Could not find pip at {pip_path}")
        sys.exit(1)

    # 3. Upgrade pip (optional but recommended)
    print("Upgrading pip inside virtual environment...")
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to upgrade pip. Proceeding with dependency installation.")

    # 4. Install requirements
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        try:
            subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
    else:
        print(f"Error: {requirements_file} not found in the root directory!")
        sys.exit(1)

    print("==================================================")
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("==================================================")
    print("To activate the virtual environment manually in your terminal:")
    if is_windows:
        print(f"  PowerShell: {activate_cmd}")
        print("  Command Prompt: .venv\\Scripts\\activate.bat")
    else:
        print(f"  Bash/Zsh: {activate_cmd}")
    print("\n*Note for VS Code / Cursor Users:*")
    print("The IDE will automatically detect the new '.venv' folder.")
    print("The 'Play' button will use this environment automatically.")
    print("==================================================")

if __name__ == "__main__":
    main()
