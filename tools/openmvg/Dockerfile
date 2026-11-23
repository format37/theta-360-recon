# OpenMVG (develop) built from source: the native spherical SfM used as the comparison run.
#   docker build -t openmvg:local tools/openmvg
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git cmake build-essential \
      libjpeg-dev libpng-dev libtiff-dev \
      libxxf86vm-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive --depth 1 --shallow-submodules -b develop https://github.com/openMVG/openMVG.git /opt/openMVG
RUN mkdir /opt/build && cd /opt/build \
    && cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/opt/openmvg \
             -DOpenMVG_BUILD_TESTS=OFF -DOpenMVG_BUILD_EXAMPLES=OFF -DOpenMVG_BUILD_DOC=OFF \
             -DOpenMVG_BUILD_GUI_SOFTWARES=OFF /opt/openMVG/src \
    && make -j"$(nproc)" install && rm -rf /opt/build
ENV PATH=/opt/openmvg/bin:$PATH
