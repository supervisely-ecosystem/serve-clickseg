FROM supervisely/serve-clickseg:1.0.12

RUN apt-get update
RUN apt-get -y install curl
RUN apt -y install wireguard iproute2
RUN apt-get -y install wget
RUN apt-get install nano

LABEL "role"="development"