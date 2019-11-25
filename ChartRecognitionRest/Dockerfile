# FROM tensorflow/tensorflow
# docker run --rm -ti chart-recognition sh
# docker build --tag chart-recognition .
# docker run --name chart-recognition -p 5000:5000 chart-recognition
FROM python:3.7-buster
COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirments.txt
CMD python PredictChartType.py
