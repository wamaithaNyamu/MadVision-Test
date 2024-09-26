from Producers.send_channel import send_to_queue
from Utilities.utilities_logging import logger
from Utilities.utilities_database  import check_value_exists,insert_data
from fastapi import FastAPI
from pydantic import BaseModel
import re



app = FastAPI()

def get_video_id(url):
  """
  This function extracts and returns the complete YouTube URL with only the video ID
  from a URL in the format "https://youtu.be/<video_id>?<query_parameters>".

  Args:
      url (str): The YouTube URL containing the video ID.

  Returns:
      str: The complete YouTube URL with only the video ID or None if the URL is not 
          in the expected format or video ID not found.
  """
  # Define a regular expression pattern to match the video ID
  pattern = r"(?:https?:\/\/)?(?:youtu\.be\/|youtube\.com\/embed\/)([^\?\&]+)"

  # Use re.search to find a match in the URL
  match = re.search(pattern, url)

  # If a match is found, return the complete URL with video ID only
  if match:
    video_id = match.group(1)
    return video_id
    # If no match is found, return None
  return None

# Define a Pydantic model for the request body
class VideoDetails(BaseModel):
      url: str

@app.post("/")
async def get_video_url(VideoDetails:VideoDetails):
    try: 
        url = VideoDetails.url
        logger.info(f"The url being processed is {url} ...")
        video_id = get_video_id(url)
        # check if in db
        is_in_db =  check_value_exists('videos','video_id',video_id)[0]['data']
        if len(is_in_db) <= 0:
            logger.info(f"The url {url} is not in the db, now adding it ...")
            # the url is not in the db
            # add it to the db
            is_added = insert_data('videos',{
                "url":url,
                "video_id":video_id
            })
            logger.info("Adding the url to the download video queue ")
            send_to_queue('download_video',{"url":url})
            return is_added
        else:
            # return clips
            logger.info("Adding the url to the download video queue ")
            send_to_queue('download_video',{"url":url})
            return {"data": f"{VideoDetails.url} has been processed "},200
        
    except Exception as e:
        logger.error(f"Something went wrong while processing the request {e}")
        return {
            "data":"something went wrong while processing the requests"
        },500