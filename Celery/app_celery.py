from celery import Celery

app = Celery('tasks',  backend='rpc://', broker='pyamqp://')

# Optional: Adjust concurrency settings
app.conf.worker_concurrency = 1  # Set to the number of concurrent tasks you want to allow
