required_packages = ["scipy", "paramiko", "h5py", "numpy", "pickle", "torch"]

def check_packages(packages):
    missing_packages = []
    for package in packages:
        try:
            __import__(package)
            print("{} is installed.".format(package))
        except ImportError:
            print("{} is NOT installed.".format(package))
            missing_packages.append(package)
    return missing_packages

if __name__ == "__main__":
    missing_packages = check_packages(required_packages)
    if missing_packages:
        print("Missing packages: {}".format(missing_packages))
    else:
        print("All required packages are installed.")