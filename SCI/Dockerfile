FROM ubuntu:18.04
RUN apt-get update && \
        apt-get -y install sudo software-properties-common \
        git wget vim apt-utils libmpfr-dev

RUN mkdir ezpc_dir
COPY EzPC/ ezpc_dir
WORKDIR /ezpc_dir
RUN ./setup_env_and_build.sh quick

RUN cd SCI && mkdir build
WORKDIR SCI/build
RUN cmake -DCMAKE_INSTALL_PREFIX=./install .. -DBUILD_TESTS=ON -DBUILD_NETWORKS=ON -DNO_REVEAL_OUTPUT=ON
RUN cmake --build . --target install --parallel

WORKDIR bin
CMD ./relu-OT r=1 & ./relu-OT r=2
