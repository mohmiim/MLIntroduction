FROM python:3.7-buster
COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirments.txt
EXPOSE 5000
CMD python ./PredictChjartType.py