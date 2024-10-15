def print_dict_structure(d, indent=0):
    """
    Recursively prints the structure of a dictionary, including nested dictionaries.
    
    Parameters:
    - d: The dictionary to print the structure of.
    - indent: The current indentation level (used for nested structures).
    """
    for key, value in d.items():
        print("  " * indent + f"- {key}: {type(value).__name__}")  # Print the key and its type
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)  # Recursively print nested dictionaries

