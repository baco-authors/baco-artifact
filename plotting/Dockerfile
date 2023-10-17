FROM python:3.8-bullseye
ENV PYTHONUNBUFFERED=1

WORKDIR /app

#RUN apt-get update
#RUN apt update --fix-missing 
#RUN apt upgrade
#RUN apt-get install -y software-properties-common gcc 
#RUN add-apt-repository -y ppa:deadsnakes/ppa

#RUN apt-get update 
#RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3.8 python3-distutils python3-pip python3-apt


COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN mkdir plots

CMD ["./make_plots.sh"]

