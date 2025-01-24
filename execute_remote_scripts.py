import paramiko # To establish an SSH connection
import sys
import os
import pickle # To store the processed data
import matplotlib.pyplot as plt
import shutil
sys.path.append(r'C:\Users\Max Tost\Desktop\Notebooks\SPC Neural Network Project')

from PasswordLac import password as pw

hostname = "lac8"
port = 22
username = "tost"
password = pw()

def execute_remote_script(script_path, remote_script_path):
    """
    Executes a Python script on the remote server and prints the output.

    Parameters:
        script_path (str): The local path to the Python script to be executed.
        remote_script_path (str): The remote path where the script will be uploaded.
    """
    try:
        # Initialize an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
        ssh_client.connect(hostname, port, username, password)  # Connect to the remote server
        
        # Open an SFTP session
        sftp = ssh_client.open_sftp()
        print("SFTP session established!")

        # Upload the script to the remote server
        sftp.put(script_path, remote_script_path)
        print(f"Uploaded {script_path} to {remote_script_path}")

        # Execute the script on the remote server
        stdin, stdout, stderr = ssh_client.exec_command(f"python {remote_script_path}")
        print(stdout.read().decode())
        print(stderr.read().decode())

        # Close the SFTP session
        sftp.close()
        ssh_client.close()
        print("SFTP session closed.")
    
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")



def execute_remote_script_download(script_path, remote_script_path, remote_folder, local_folder):
    """
    Executes a Python script on the remote server and retrieves the contents of a folder.

    Parameters:
        script_path (str): The local path to the Python script to be executed.
        remote_script_path (str): The remote path where the script will be uploaded.
        remote_folder (str): The remote folder to be downloaded.
        local_folder (str): The local folder where the contents will be saved.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """

    try:
        # Initialize an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
        ssh_client.connect(hostname, port, username, password)  # Connect to the remote server
        
        # Open an SFTP session
        sftp = ssh_client.open_sftp()
        print("SFTP session established!")

        # Upload the script to the remote server
        sftp.put(script_path, remote_script_path)
        print("Uploaded {} to {}".format(script_path, remote_script_path))

        # Execute the script on the remote server
        stdin, stdout, stderr = ssh_client.exec_command("python {}".format(remote_script_path))
        print(stdout.read().decode())
        print(stderr.read().decode())

        # Ensure the local folder exists
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        # Download the contents of the remote folder
        print(f"Downloading contents of remote folder {remote_folder} to local folder {local_folder}")
        for file_attr in sftp.listdir_attr(remote_folder):
            remote_file_path = os.path.join(remote_folder, file_attr.filename).replace('\\', '/')
            local_file_path = os.path.join(local_folder, file_attr.filename)

            print(f"Downloading {remote_file_path} to {local_file_path}")
            sftp.get(remote_file_path, local_file_path)

        print(f"Downloaded all files from {remote_folder} to {local_folder}")

        # Close the SFTP session
        sftp.close()
        ssh_client.close()
        print("SFTP session closed.")
    
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred: {}".format(e))



def load_pickle(file_name):
    """
    Loads data from a Pickle file.

    Parameters:
        file_name (str): The name of the Pickle file.

    Returns:
        dict: The loaded data.
    """
    with open(file_name, 'rb') as pickle_file:
        data = pickle.load(pickle_file, encoding='latin1')
    return data

def plot_data(data, key):
    """
    Plots the signal and time data for a given key.

    Parameters:
        data (dict): The processed data.
        key (str): The key to plot.
    """
    signal = data[key]['signal']
    time = data[key]['time']

    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.title(f"{key} Signal vs Time")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.grid(True)
    plt.show()

def download_existing_pickle(remote_file_path, local_file_path):
    """
    Downloads an existing Pickle file from the remote server to the local machine.

    Parameters:
        remote_file_path (str): The path to the Pickle file on the remote server.
        local_file_path (str): The path to save the Pickle file on the local machine.
    """
    try:
        # Initialize an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown host keys
        ssh_client.connect(hostname, port, username, password)  # Connect to the remote server
        
        # Open an SFTP session
        sftp = ssh_client.open_sftp()
        print("SFTP session established!")

        # Download the Pickle file
        print("Attempting to download {} to {}".format(remote_file_path, local_file_path))
        sftp.get(remote_file_path, local_file_path)
        print("Downloaded {} to {}".format(remote_file_path, local_file_path))

        # Close the SFTP session
        sftp.close()
        ssh_client.close()
        print("SFTP session closed.")
    
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred: {}".format(e))
