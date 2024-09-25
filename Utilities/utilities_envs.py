
import os
from dotenv import load_dotenv
# Load the environment variables from .env file
load_dotenv()

# Access the variables
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
videos_output= os.getenv('VIDEOS_FOLDER_NAME')
clips_output= os.getenv('CLIPS_FOLDER_NAME')
Supabase_Service_Key=os.getenv('Supabase_Service_Key')

