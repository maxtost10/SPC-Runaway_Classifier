import paramiko
import sys
import os
import pickle
import matplotlib.pyplot as plt
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



def execute_remote_script_download(script_path, remote_script_path, result_file):
    """
    Executes a Python script on the remote server and retrieves the result file.

    Parameters:
        script_path (str): The local path to the Python script to be executed.
        remote_script_path (str): The remote path where the script will be uploaded.
        result_file (str): The name of the result file to be retrieved.

    Returns:
        dict: The contents of the result file as a Python dictionary, or None if an error occurs.
    """
    result_data = None

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

        # Download the result file
        local_result_path = os.path.join(os.path.dirname(script_path), result_file)
        remote_result_path = os.path.join(os.path.dirname(remote_script_path), result_file).replace('\\', '/')
        print("Attempting to download {} to {}".format(remote_result_path, local_result_path))
        sftp.get(remote_result_path, local_result_path)
        print("Downloaded {} to {}".format(remote_result_path, local_result_path))

        # Load the result data
        with open(local_result_path, 'rb') as pickle_file:
            result_data = pickle.load(pickle_file, encoding='latin1')

        # Close the SFTP session
        sftp.close()
        ssh_client.close()
        print("SFTP session closed.")
    
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred: {}".format(e))

    return result_data



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
