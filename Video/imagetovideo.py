import cv2
import os
import moviepy.editor as moviepy

image_folder = 'C:\\Users\\mans3851\\OneDrive - Nexus365\\phd\\Year 1\\3DEC test\\Validation_tests\\Validation\\DELFT\\Long wall\\New stage\\Cyclic\\1'
video_name = 'video'
video_name_temp = os.path.join(image_folder, video_name + '.' + 'avi')
video_name = os.path.join(image_folder, video_name + '.' + 'mp4')

# Convert images into video
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name_temp, 0, 30, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()

# Convert avi files into mp4 files
clip = moviepy.VideoFileClip(video_name_temp)
clip.write_videofile(video_name)

# Delete the avi file
os.remove(video_name_temp)

# Delete the images
for img in os.listdir(image_folder):
    if img.endswith(".png"):
        os.remove(os.path.join(image_folder, img))
