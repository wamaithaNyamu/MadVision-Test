from Producers.send_channel import send_to_queue
from Utilities.utilities_logging import logger
from Utilities.utilities_database  import check_value_exists,insert_data
from fastapi import FastAPI
from pydantic import BaseModel




app = FastAPI()

# Define a Pydantic model for the request body
class VideoDetails(BaseModel):
      url: str

@app.post("/")
async def get_video_url(VideoDetails:VideoDetails):
    try: 
        url = VideoDetails.url
        logger.info(f"The url being processed is {url} ...")
        
        # check if in db
        is_in_db =  check_value_exists('videos','url',url)[0]['data']
        if len(is_in_db) <= 0:
            logger.info(f"The url {url} is not in the db, now adding it ...")
            # the url is not in the db
            # add it to the db
            is_added = insert_data('videos',{
                "url":url
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