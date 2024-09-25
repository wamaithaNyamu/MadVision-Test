import json

def bytes_to_dict(byte_data):
    # Decode the bytes into a string (assumes UTF-8 encoding)
    json_str = byte_data.decode('utf-8')

    # Parse the JSON string into a dictionary
    dictionary = json.loads(json_str)

    return dictionary

def dict_to_bytes(dictionary):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(dictionary)

    # Encode the JSON string into bytes (using UTF-8 encoding)
    byte_data = json_str.encode('utf-8')

    return byte_data