# This file contains all the code for database operation:
#     1: Uploading to db (POST)
#     2. Updating the db (PUT)
#     3. Getting data from the db (GET)
#     4. Getting specific data from the db (GET)
#     5. Deleting from the db (DELETE)


from supabase import create_client, Client
from Utilities.utilities_logging import logger
from Utilities.utilities_envs import supabase_key, supabase_url,Supabase_Service_Key


supabase: Client = create_client(supabase_url, Supabase_Service_Key)


# Function to check if a column exists in the table
def column_exists(table, column_name):
    try:
        # Get the schema for the table
        schema = supabase.table(table).get_schema()
        # Check if the column exists
        return column_name in schema['columns']
    except Exception as e:
        logger.error(f"Error checking column existence for {column_name} in {table}: {e}")
        return False
    

# Function to check if the value exists in a column
def check_value_exists(table, column_name, desired_value):
    try:
        # Query the Supabase database
        result = supabase.table(table).select(column_name).eq(column_name, desired_value).limit(1).execute()

        # Check if any data was returned
        if result.data:
            logger.info(f"Value {desired_value} exists in {table} under column {column_name}.")
            return {
                'success': True,
                'exists': True,
                'data': result.data
            }, 200
        else:
            logger.warning(f"Value {desired_value} does not exist in {table} under column {column_name}.")
            return {
                'success': True,
                'exists': False,
                'data': []
            }, 404  # Not Found

    except Exception as e:
        logger.error(f"There was an error checking {desired_value} from the {table} table, column {column_name}. The error is {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500  # Internal Server Error


# Function to insert data into a specified table
def insert_data(table, data):
    try:
        # Perform the insert operation
        response =  supabase.table(table).insert(data).execute()
        data = list(response)[0][1]
        logger.info(f"Data inserted successfully into {table}: {data}")
        return {
            'success': True,
            'data':data
        }, 200
    
    except Exception as e:
        logger.error(f"There was an error inserting data into {table}. The error is {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500




def update_data(table, data, condition):
    try:
        logger.info(f"The condition is of type {type(condition)} and its {condition['url']} ")
        # Perform the update operation
        response = supabase.table(table).update(data).eq('url' , condition['url']).execute()
        logger.info(f"Here is the update response {response}")
   

        updated_data = response.data  # Response contains updated data

        logger.info(f"Data inserted successfully into {table}: {data}")
        return {
            'success': True,
            'data':data
        }, 200
    

    except Exception as e:
        logger.error(f"There was an error updating data in {table}. The error is {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500  # Internal Server Error        



# Function to upload a file to a Supabase bucket
def upload_file(bucket_name, file_path, file_name):
    try:
        # Open the file in binary mode
        with open(file_path, 'rb') as file:
            logger.info("Uploading to supabase ...")
            # Upload the file to the specified bucket
            response = supabase.storage.from_(bucket_name).upload(file_name, file)

            logger.info(f"Here is the upload file response {response}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {
            'success': False,
            'error': "File not found."
        }, 404  # Not Found
    except Exception as e:
        logger.error(f"There was an error uploading file to bucket '{bucket_name}'. The error is {e}")
        return {
            'success': False,
            'error':"There was an error uploading file to bucket"
        }, 500  # Internal Server Error
        
        
        
        
def check_value_exists_in_column(table, url, column_name):
    try:
        # Query the Supabase database
        result = supabase.table(table).select(column_name).eq('url', url).execute()

        # Check if any data was returned and the transcription value is not None
        if result.data and result.data[0][column_name] is not None:
            logger.info(f"Value exists in {table} under column {column_name} for URL {url}.")
            logger.info(result.data[0][column_name])
            return {
                'success': True,
                'exists': True,
                'data': result.data[0][column_name]
            }, 200
        else:
            logger.warning(f"Value does not exist or is None in {table} under column {column_name} for URL {url}.")
            return {
                'success': True,
                'exists': False,
                'data': []
            }, 404  # Not Found

    except Exception as e:
        logger.error(f"There was an error checking {column_name} from the {table} table for URL {url}. The error is {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500  # Internal Server Error
