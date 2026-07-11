FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG ROS_DISTRO=jazzy
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl software-properties-common \
    && add-apt-repository universe \
    && ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p')" \
    && test -n "${ROS_APT_SOURCE_VERSION}" \
    && curl -fsSL -o /tmp/ros2-apt-source.deb \
       "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-${ROS_DISTRO}-ros-base \
       ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/* /tmp/ros2-apt-source.deb

WORKDIR /app
COPY python-package /app/python-package
COPY examples/team_face_recognition /app/team_face_recognition

RUN python3 -m pip install --no-cache-dir \
      onnx onnxruntime opencv-python-headless requests scipy scikit-image tqdm \
    && python3 -m pip install --no-cache-dir --no-deps /app/python-package \
    && chmod +x /app/team_face_recognition/deploy/entrypoint.sh

ENV ROS_DISTRO=${ROS_DISTRO}
ENTRYPOINT ["/app/team_face_recognition/deploy/entrypoint.sh"]
CMD ["python3", "/app/team_face_recognition/g1_face_bridge.py"]
