FROM python:3.14

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -e .

COPY . /app


EXPOSE 5001

CMD ["python", "app/application.py"]

