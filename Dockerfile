FROM ubuntu:20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
git \
python3-pip \
libgtk2.0-dev \
&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/me-X-us/convert-motion-mask.git
WORKDIR /convert-motion-mask
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["./server.py"]
