from scipy.io import loadmat
import sys
import os
import paramiko
sys.path.append(r'C:\Users\Max Tost\Desktop\Notebooks\SPC Neural Network Project')

from PasswordLac import password as pw

hostname = "lac8"
port = 22
username = "tost"
password = pw()

def load_mat_file(file_name, local_path, delete=True):
    """
    Downloads a .mat file from a remote server using SFTP, loads its contents, 
    and removes the local copy after loading to save storage.

    Parameters:
        file_name (str): The name of the .mat file to be downloaded.
        local_path (str): The local directory where the file will be temporarily stored.

    Returns:
        dict: The contents of the .mat file as a Python dictionary, or None if an error occurs.
    """
    mat_contents = None  # Initialize the variable to store the .mat file contents

    try:
        # Initialize an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
        ssh_client.connect(hostname, port, username, password)  # Connect to the remote server
        
        # Open an SFTP session
        sftp = ssh_client.open_sftp()
        print("SFTP session established (read-only)!")

        # Define a function to disable write operations for this session
        def read_only_error(*args, **kwargs):
            raise PermissionError("This is a read-only session. Write operations are disabled.")
        
        # Overwrite SFTP write methods with the read-only function
        sftp.put = read_only_error
        sftp.remove = read_only_error
        sftp.rename = read_only_error

        # Construct remote and local file paths
        remote_file_path = r"/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/" + file_name  # Remote file location
        local_file_path = local_path + "\\" + file_name  # Temporary local storage location

        # Check if the file exists on the server
        try:
            sftp.stat(remote_file_path)  # Check if the file exists (raises an exception if it doesn't)
            print(f"File '{remote_file_path}' exists on the server.")
            
            # Download the file to the local directory
            print(f"Downloading '{remote_file_path}' to '{local_file_path}'...")
            sftp.get(remote_file_path, local_file_path)  # Perform the download
            print("Download complete. Begin loading the file locally.")

            # Load the downloaded .mat file into a dictionary
            mat_contents = loadmat(local_file_path)  # Read the .mat file
            print("MAT File Contents:")
            for key, value in mat_contents.items():
                # Skip internal MATLAB metadata keys (e.g., '__header__', '__version__')
                if not key.startswith("__"):
                    print(f"{key}: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            if delete:
                # Remove the downloaded file to save disk space
                print(f"Deleting {local_file_path}. Variable remains.")
                os.remove(local_file_path)

        except FileNotFoundError:
            # Handle the case where the file doesn't exist on the server
            print(f"File '{remote_file_path}' does not exist on the server.")

        # Close the SFTP session
        sftp.close()
        ssh_client.close()
        print("SFTP session closed.")
    
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")

    # Return the loaded .mat file contents (or None if an error occurred)
    return mat_contents


def list_remote_files():
    """
    Connects to the remote server and returns a list of files in the target directory.

    Returns:
        list: A list of file names in the remote directory, or an empty list if an error occurs.
    """
    files = []
    try:
        # Initialize an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
        ssh_client.connect(hostname, port, username, password)  # Connect to the remote server
        
        # Open an SFTP session
        sftp = ssh_client.open_sftp()
        print("SFTP session established (read-only)!")

        # Define a function to disable write operations for this session
        def read_only_error(*args, **kwargs):
            raise PermissionError("This is a read-only session. Write operations are disabled.")
        
        # Overwrite SFTP write methods with the read-only function
        sftp.put = read_only_error
        sftp.remove = read_only_error
        sftp.rename = read_only_error

        # Define the target directory path
        remote_path = "/Lac8_D/DEFUSE/DEFUSE_DB/DB_mat/"
        
        # List the files in the directory
        files = sftp.listdir(remote_path)
        print(f"Files in '{remote_path}':")
        for file in files:
            print(file)

        # Close the SFTP session
        sftp.close()
        ssh_client.close()
        print("SFTP session closed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return files

