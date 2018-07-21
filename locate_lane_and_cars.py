#!/usr/bin/env python
#
# Identify and draw our lane and expected path of travel.
# Identify and frame nearby vehicles.

from moviepy.editor import VideoFileClip

from zone import LaneBoundaryZone, VehicleCollisionZone

import lesson_functions

def lane_car_locate_pipeline(vehicle_zone, lane_zone):
    """Return a function that takes an image and runs the pipeline identifying lane boundaries and nearby cars."""
    def _lane_car_locate_pipeline(rgb_img):
        """Run both vehicle_zone and lane_zone pipelines."""
        car_matches = vehicle_zone.locate_nearby_cars(rgb_img)
        lane_img = lane_zone.locate_lane_bounds(rgb_img)
        return lesson_functions.draw_boxes(lane_img, car_matches)
    return _lane_car_locate_pipeline

def main():
    """Start here..."""
    # Create a frane iterator of the project video to run our pipeline on each frame.
    project_video = VideoFileClip('./project_video.mp4')

    # Determine dimensions of the video.
    sample_img_shape = project_video.get_frame(0).shape

    # Create the pipeline function.
    vehicle_zone = VehicleCollisionZone()
    lane_zone = LaneBoundaryZone(sample_img_shape[0], sample_img_shape[1])
    pipeline_fn = lane_car_locate_pipeline(vehicle_zone, lane_zone)

    # Pass the pipeline function to the iterator and fire it up saving the resulting stream to a video file.
    project_video_pipeline = project_video.fl_image(pipeline_fn)
    project_video_pipeline.write_videofile('./output_videos/project_video_pipeline.mp4', audio=False)

if __name__ == '__main__':
    main()
