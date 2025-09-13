FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=rag_django.settings
RUN python manage.py migrate --no-input || true
CMD ["gunicorn", "rag_django.wsgi", "--bind", "0.0.0.0:8000"]
