import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

if __name__ == "__main__":
    packages_to_install = ["paramiko", "h5py"]
    for package in packages_to_install:
        try:
            install_package(package)
            print("{} installed successfully.".format(package))
        except Exception as e:
            print("Failed to install {}: {}".format(package, e))