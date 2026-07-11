#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/noetic/setup.bash

exec roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  color_width:=640 \
  color_height:=480 \
  color_fps:=15 \
  depth_width:=640 \
  depth_height:=480 \
  depth_fps:=15
