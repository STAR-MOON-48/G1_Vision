#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/noetic/setup.bash
unset ROS_DISTRO
source /opt/ros/foxy/setup.bash

OVERLAY="${HOME}/ros1_bridge_overlay/opt/ros/foxy"
export AMENT_PREFIX_PATH="${OVERLAY}:/opt/ros/foxy"
export LD_LIBRARY_PATH="${OVERLAY}/lib:/opt/ros/noetic/lib:/opt/ros/foxy/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

exec "${OVERLAY}/lib/ros1_bridge/dynamic_bridge" --ros-args --disable-rosout-logs
