./redis.sh
celery -A capp.celery worker --loglevel=info
python3 capp.py
