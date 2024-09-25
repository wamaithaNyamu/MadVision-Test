# MADVISION

# Backend

Install dependencies
```sh
pip install requirements.txt
```

Install ffmpeg (MacOS)
```
brew install ffmpeg
```

Run rabbit mq from docker
```sh
sudo docker compose up
```
Give permissions

```sh
chmod +x run_consumer.py
```
Run consumers
```sh
./run_consumers.py
```

Run celery worker

Give permissions 

```shell
chmod +x run_celery.py

```

Run on another terminal
```
./run_celery

```

Run Fast API in another terminal window

```shell
fastapi dev app.py
```

Navigate to https://localhost:8000/docs to interact with the API


