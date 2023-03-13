import moviepy.editor as moviepy
clip = moviepy.VideoFileClip("video.avi")
clip.write_videofile("video.mp4")