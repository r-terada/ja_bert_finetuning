FROM python:3.7
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    bzip2 \
    libjuman \
    libcdb-dev \
    libboost-all-dev \
    make \
    cmake \
    zlib1g-dev


# Install compilers.
RUN apt-get install -y gcc && \
    apt-get install -y g++


RUN pip install --upgrade pip

WORKDIR /app

RUN curl -L -o jumanpp-2.0.0-rc2.tar.xz https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc2/jumanpp-2.0.0-rc2.tar.xz && \
    tar Jxfv jumanpp-2.0.0-rc2.tar.xz && \
    cd jumanpp-2.0.0-rc2/ && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ && \
    make && \
    make install

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/megagonlabs/jrte-corpus