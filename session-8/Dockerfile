# docker run --rm -ti gan sh
# docker build --tag gan .
# docker run -p 5000:5000 gan
FROM python:3.7-buster
COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirments.txt
CMD python GenerateFlowerRest.py
