FROM supervisely/base-py-sdk:6.69.70

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y git ffmpeg libsm6 libxext6 ninja-build

RUN pip3 install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
RUN pip3 install mmcv-full==1.4.1

RUN pip3 install \
    Pillow==8.4.0 \
    Cython==0.29.23 \
    attr==0.3.1 \
    tensorboard==2.12.0 \
    termcolor==1.1.0 \
    timm==0.3.2 \
    easydict==1.9 \
    albumentations==0.5.1 \
    thop \
    gdown

RUN pip3 install supervisely==6.73.102

RUN mkdir /checkpoints
RUN gdown https://drive.google.com/file/d/1DkFun_tiw7z7RpjDtwqV65k1e9jxnLkr/view?usp=share_link --fuzzy -O /checkpoints/segformerb3-s2_9.pth
