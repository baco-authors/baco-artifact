FROM python:3.8-bullseye
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN mkdir plots

CMD ["./make_plots.sh"]

