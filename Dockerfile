FROM supervisely/serve-clickseg:1.0.2

RUN pip install git+https://github.com/supervisely/supervisely.git@bbox_tracking_debug

WORKDIR /app
COPY . /app

EXPOSE 80

ENV APP_MODE=production ENV=production

ENTRYPOINT ["python", "-u", "-m", "uvicorn", "src.main:m.app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
