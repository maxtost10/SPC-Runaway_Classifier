def print_dict_structure(d, max_layers=-1, indent=0):
    """
    Recursively prints the structure of a dictionary, including nested dictionaries.
    
    Parameters:
    - d: The dictionary to print the structure of.
    - indent: The current indentation level (used for nested structures).
    - max_layers: int; the depth of the deepest layer that should be printed
    
    Example:
        print_dict_structure(d, 2); prints the first two layers of the dictionary d
        
    """
    for key, value in d.items():
        if indent == max_layers:
            break
        print("  " * indent + f"- {key}: {type(value).__name__}")  # Print the key and its type
        if isinstance(value, dict):
            print_dict_structure(value, max_layers, indent + 1)  # Recursively print nested dictionaries

            
            
            
            
import json

def load_json(path, print_ = False):
    """
    Converts json data to a dictionary
    
    Args:
        path: string; Path to json file
        print_ : boolean; If the structure of the json file shall be printed
    
    Returns:
        disruption_data: dictionary (or sometimes list) of the json file
        
    Example:
        json_dict = load_json(path)
    
    """
    # Load JSON data
    with open(path) as json_file:
        disruption_data = json.load(json_file)
        
    if print_ == True:
        print('The structure of the json file is\n')
        print_dict_structure(disruption_data, max_layers=-1, indent=0)


    return disruption_data