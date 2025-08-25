FROM deepprinciple-cn-beijing.cr.volces.com/public-registry/python:3.10.16

ARG USE_PROXY=true
ARG PROXY_SERVER
ENV http_proxy ${PROXY_SERVER}
ENV https_proxy ${PROXY_SERVER}

ENV HTTP_PROXY ${PROXY_SERVER}
ENV HTTPS_PROXY ${PROXY_SERVER}

RUN echo $USE_PROXY

# If USE_PROXY is false, unset the proxy environment variables 
RUN if [ "$USE_PROXY" = "false" ]; then unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; fi

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir torch -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir torch_geometric -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir pytorch-lightning -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir torchdiffeq -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN apt-get update && apt-get -y install build-essential python3-dev python3-pip libomp-dev
RUN pip install torch_scatter -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY