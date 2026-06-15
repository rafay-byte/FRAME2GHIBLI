ffmpeg -framerate 12 -i output_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
