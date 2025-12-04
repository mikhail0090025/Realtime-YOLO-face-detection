FROM my-base-image:latest

WORKDIR /app

RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ./model /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
