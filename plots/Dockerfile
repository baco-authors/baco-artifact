FROM ubuntu
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y python3.8 python3-distutils python3-pip python3-apt


COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["./run.sh"]

