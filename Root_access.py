import os
import subprocess

def grant_root_access():
    try:
        # Check if script is already running as root
        if os.geteuid() == 0:
            print("Already running as root.")
        else:
            # Use sudo to elevate privileges
            subprocess.run(["sudo", "python3", __file__])
            print("Root access granted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    grant_root_access()
