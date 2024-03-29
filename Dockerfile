FROM supervisely/serve-clickseg:1.0.2

RUN pip install git+https://github.com/supervisely/supervisely.git@bbox_tracking_debug

COPY . /repo

EXPOSE 80

ENV PYTHONPATH="${PYTHONPATH}:/repo/ClickSEG"
ENV APP_MODE=production ENV=production

ENTRYPOINT ["python3", "-u", "-m", "uvicorn", "src.main:m.app", "--app-dir", "repo"]
CMD ["--host", "0.0.0.0", "--port", "80"]
