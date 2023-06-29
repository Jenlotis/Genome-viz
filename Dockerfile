FROM python:3.9
ADD Dashboard ./Dashboard/
ADD requirements.txt ./
RUN pip install -r requirements.txt
WORKDIR /Dashboard
EXPOSE 8050
CMD ["python", "./app.py"]
