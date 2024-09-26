import re
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import cv2
import yt_dlp
import json
import time
import requests
import torch
import whisper
from moviepy.audio.fx.all import audio_normalize
from moviepy.editor import VideoFileClip
from Utilities.utilities_file_manipulations import create_directory
from Utilities.utilities_logging import logger
from Utilities.utilities_envs import videos_output,clips_output
from Utilities.utilities_database import  check_value_exists_in_column, update_data, upload_file

from Celery.app_celery import app
import threading



device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)
clips_output = create_directory(clips_output)

videos_output = create_directory(videos_output)

print("videos_output ",videos_output)
# yt-dlp options
# yt-dlp options
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(videos_output, '%(id)s.%(ext)s'),
    'noplaylist': True,
    'progress_hooks': [lambda d: logger.info(f"Downloading {d['filename']}: {d['downloaded_bytes']} bytes downloaded")],
    'postprocessors': [{
        'key': 'FFmpegVideoConvertor',
        'preferedformat': 'mp4',  # Convert to mp4 after download
    }],
    'continue_dl': True  # This option allows resuming downloads
}

def delete_temp_files(directory):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Check if 'TEMP' is in the filename
        if 'TEMP' in filename:
            file_path = os.path.join(directory, filename)
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def auto_reframe(clip, target_aspect_ratio):
    def detect_face(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return faces[0]
        return None

    w, h = clip.size
    if w / h > target_aspect_ratio:
        new_w = int(h * target_aspect_ratio)
        frame = clip.get_frame(0)
        face = detect_face(frame)
        if face is not None:
            x, y, fw, fh = face
            center_x = x + fw // 2
            x1 = max(0, min(w - new_w, center_x - new_w // 2))
        else:
            x1 = (w - new_w) // 2
        return clip.crop(x1=x1, y1=0, x2=x1+new_w, y2=h)
    else:
        new_h = int(w / target_aspect_ratio)
        y1 = (h - new_h) // 2
        return clip.crop(x1=0, y1=y1, x2=w, y2=y1+new_h)
    
def extract_and_enhance_clip(video_path, start_time, end_time, output_path):
    try:
        # Load the video
        clip = VideoFileClip(video_path)
        # Ensure start and end times are within the video duration
        video_duration = clip.duration
        logger.info(f"Video duration: {video_duration} seconds")
        
        if start_time >= video_duration:
            logger.warning(f"Start time {start_time} exceeds video duration {video_duration}")
            return False
        
        if end_time > video_duration:
            logger.warning(f"End time {end_time} exceeds video duration {video_duration}. Clipping to {video_duration}")
            end_time = video_duration
        
        if start_time >= end_time:
            logger.warning(f"Start time {start_time} is greater than or equal to end time {end_time}")
            return False
        logger.info("Making subclip")
        # Extract the clip within the valid duration
        subclip = clip.subclip(start_time, end_time)
          
          # Check the duration of the subclip
        subclip_duration = subclip.duration
        logger.info(f"The subclip has a duration of {subclip_duration}")
        
        if subclip_duration > 180:  # 180 seconds is 3 minutes
            logger.warning(f"Subclip duration {subclip_duration} exceeds 3 minutes. Skipping extraction.")
            return False
        logger.info("Auto reframing")
        # Auto reframe clip
        subclip = auto_reframe(subclip, 1)  # For 1:1 aspect ratio
        logger.info("fade in")
        # Add fade in and fade out
        subclip = subclip.fadein(0.5).fadeout(0.5)
        logger.info("Normalizing audio")
        # Normalize the audio
        subclip = subclip.fx(audio_normalize)
        
        logger.info(f"saving to file...")
        # Write the final clip
        def save_video():
            start_time = time.time()
            subclip.write_videofile(output_path,
                                codec="libx264",
                                audio_codec="aac",
                                logger=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"],
                                threads=4,
                                preset='superfast',  
                                 )
            end_time = time.time()
            logger.info(f"Processing time: {end_time - start_time:.2f} seconds")

        thread = threading.Thread(target=save_video)
        thread.start()
        thread.join()  # Wait for the thread to complete
        logger.info(f"Clip extracted and enhanced successfully: {output_path}")
        return True
 
    except Exception as e:
        logger.error(f"Error extracting and enhancing clip: {e}")
    finally:
        clip.close()  # Ensure the original clip is properly closed



def find_insightful_clips(transcript, content_type, min_duration=60, max_duration=180, min_clips=2, max_clips=8):
    
    prompt = f"""
    Analyze the following {content_type} transcript and identify between {min_clips} and {max_clips} of the most valuable or interesting sections. These sections should have potential to engage viewers, even if they're not all equally insightful. For each section:

    1. Ensure it starts and ends at natural breaks in speech, preferably at the beginning of a new thought or topic.
    2. Choose sections that can stand alone as engaging content.
    3. Look for sections that contain any of the following:
       - Key insights or unique perspectives
       - Interesting facts or ideas
       - Practical advice or information
       - Emotionally engaging or inspiring moments
       - Points that might spark curiosity or discussion
    4. Pay attention to the start and end of each clip:
       - Start the clip with a complete sentence or thought that introduces the topic.
       - End the clip with a concluding statement or a natural pause in the conversation.

    IMPORTANT: Always select at least {min_clips} clips, even if they don't seem highly insightful. Choose the best available options.

    Return a JSON array of clips, where each clip is an object with 'start' and 'end' times in seconds, a 'summary' field, and a 'relevance_score' field (0-100).
    The 'relevance_score' should indicate how engaging or valuable the clip is, with 100 being the highest quality.
    Ensure each clip is between {min_duration} and {max_duration} seconds long.
    Vary the clip lengths within the allowed range for diversity.

    Transcript:
    {transcript}
    """

    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are an AI assistant that analyzes video transcripts to find engaging and potentially valuable content."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8,
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad responses
        result = response.json()
        content = result['choices'][0]['message']['content']
        logger.info(f"API Response: {content}")

        try:
            clips = json.loads(content)
            # Sort clips by relevance_score in descending order
            clips.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Ensure we have at least min_clips
            if len(clips) < min_clips:
                logger.warning(f"Only {len(clips)} clips found. This is less than the minimum of {min_clips}.")
                # If we have less than min_clips, we'll create additional clips by splitting the transcript
                total_duration = sum(clip['end'] - clip['start'] for clip in clips)
                remaining_duration = len(transcript.split()) / 2  # Rough estimate of duration based on word count
                additional_clips_needed = min_clips - len(clips)
                chunk_duration = remaining_duration / additional_clips_needed

                for i in range(additional_clips_needed):
                    start_time = total_duration + (i * chunk_duration)
                    end_time = start_time + chunk_duration
                    clips.append({
                        'start': start_time,
                        'end': end_time,
                        'summary': f"Additional clip {i+1} to meet minimum clip requirement",
                        'relevance_score': 50  # Neutral score for additional clips
                    })

            # Select top clips, ensuring we have at least min_clips and at most max_clips
            selected_clips = clips[:max_clips]
            return selected_clips
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            logger.error(f"Problematic content: {content}")
            # If JSON parsing fails, create default clips
            return [
                {
                    'start': 0,
                    'end': min_duration,
                    'summary': "Default clip 1 due to parsing error",
                    'relevance_score': 50
                },
                {
                    'start': min_duration,
                    'end': 2 * min_duration,
                    'summary': "Default clip 2 due to parsing error",
                    'relevance_score': 50
                }
            ]
    except Exception as e:
        logger.error(f"Error in API call: {e}")
        # If API call fails, create default clips
        return [
            {
                'start': 0,
                'end': min_duration,
                'summary': "Default clip 1 due to API error",
                'relevance_score': 50
            },
            {
                'start': min_duration,
                'end': 2 * min_duration,
                'summary': "Default clip 2 due to API error",
                'relevance_score': 50
            }
        ]


def identify_content_type(transcript):
    prompt = f"Analyze the following transcript and identify the type of content (e.g., podcast, interview, educational video, etc.):\n\n{transcript[:1000]}"
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an AI that identifies types of video content."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()  # Raise an error for bad responses
    result = response.json()
    return result['choices'][0]['message']['content']

def string_to_json(json_string):
    try:
        # Try to parse the string into JSON
        json_data = json.loads(json_string)
        logger.info("Valid JSON string.")
        return json_data
    except json.JSONDecodeError as e:
        # Handle the case where the string is not valid JSON
        logger.error(f"Invalid JSON string: {e}")
        return None

def save_transcription_to_json(transcription, output_json_path):
    try:
        # Prepare the data
        data = {
            "transcription": transcription
        }

        # Write the data to a JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)  # Use indent for pretty printing

        logger.info(f"The transcription was saved to {output_json_path}")

    except Exception as e:
        logger.error(f"Error saving transcription to JSON: {e}")
        
        
def transcribe_with_whisper(video_path, video_id):
    try:
        logger.info("Check if info is in db")
        is_transcribed = check_value_exists_in_column('videos', video_id, 'transcription')

        # Proceed with transcription only if not already transcribed
        if not is_transcribed[0]['exists']:
            logger.info(f"Starting transcription for {video_path}")
            
            # Ensure video_path is a string
            if isinstance(video_path, bytes):
                video_path = video_path.decode('utf-8')  # Convert bytes to string if necessary
            
            transcription = whisper_model.transcribe(video_path, fp16=False)
            logger.info(f"The transcription was {transcription}")
            update_data('videos', {"transcription": transcription}, {"video_id": video_id})
        else:
            transcription = is_transcribed[0]['data']
        
        # Ensure transcription contains a 'text' field
        if 'text' not in transcription:
            raise ValueError("The transcription data does not contain a 'text' field.")
        
        logger.info(f"Identifying content type for {video_path}")
        is_content_type_identified = check_value_exists_in_column('videos', video_id, 'content_type')
        
        if not is_content_type_identified[0]['exists']:
            content_type = identify_content_type(transcription['text'])
            update_data('videos', {"content_type": content_type}, {"video_id": video_id})
        else:
            content_type = is_content_type_identified[0]['data']
        
        logger.info(f"Identifying insightful clips for {video_path}")
        is_contentinsightful = check_value_exists_in_column('videos', video_id, 'clips')
        
        if not is_contentinsightful[0]['exists']:
            logger.info(f"Finding insightful clips for {video_path}")
            clips = find_insightful_clips(transcription['text'], content_type)
            update_data('videos', {"clips": clips}, {"video_id": video_id})
        else:
            clips = is_contentinsightful[0]['data']
            
            
        logger.info(f"Processing clips: {clips}")
        processed_clips = []
        for i, clip in enumerate(clips):
            logger.info(f"Extracting and enhancing clip {i+1} for {video_path}")
            filename = f"{video_id}_clip_{i+1}.mp4"
            output_path = os.path.join(clips_output, filename)
            
            if os.path.exists(output_path):
                logger.info(f"File {filename} already exists. Skipping clip {i+1}.")
                continue
            logger.info(f"The before time and clip is {clip}")
            # Convert clip start and end times to integers
            start_time = int(clip['start'])
            end_time = int(clip['end'])
            logger.info(f"The start time was {start_time} and the end time was {end_time}")
            logger.info(f"The transcription was  of type {type(transcription)}")
            clip_details = extract_and_enhance_clip(video_path, start_time, end_time, output_path)

            if clip_details:
                # Ensure indices for slicing are integers
                    logger.info(f"The clip details extracted were: {clip_details}")
                    if  isinstance(transcription, str):
                        transcription = string_to_json(transcription)
                        logger.info(f"The transcription transformed was  of type {type(transcription)}")
                    logger.info(f"This would upload to supabase if it was the pro plan for a limit of 50mb")
                    # upload_file("clips_generated", output_path, filename)
                    processed_clips.append({
                        'path': output_path,
                        'transcript': transcription['text'],
                        'start': start_time,
                        'end': end_time
                    })
            else:
                continue
            
        delete_temp_files(".")

        return processed_clips
    except Exception as e:
        logger.error(f"Error in process_video for {video_path}: {str(e)}")
        return []



@app.task
def process_video(url):
    logger.info(f"Starting the video download using celery {url}")
    logger.info(f"The type of data is {type(url)}")
    url = url['url']
    logger.info(f"The url is {url}")
    
    # Check if the video already exists
    if os.path.exists(url):
                # Regex pattern
        pattern = r"(?:https?:\/\/)?(?:[0-9A-Z-]+\.)?(?:youtube|youtu|youtube-nocookie)\.(?:com|be)\/(?:watch\?v=|watch\?.+&v=|embed\/|v\/|.+\?v=)?([^&=\n%\?]{11})"

        # Match the video ID
        match = re.search(pattern, url)

        if match:
             video_id = match.group(1)
             print(f"Video ID: {video_id}")
        else:
              print("Video ID not found in the URL.")
        logger.info(f"The video already exists at {url}. Adding to the transcribe queue.")
        processed_clips = transcribe_with_whisper(url,video_id)
        logger.info(f"Processed clips are : {len(processed_clips)}")

        return
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            path = os.path.join(videos_output, f"{info['id']}.mp4"), info['id']
            logger.info(f"The video was saved to {path}")
            
            # send the local path to the transcribe queue 
            transcribe_with_whisper(path[0],path[1])
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None, None



