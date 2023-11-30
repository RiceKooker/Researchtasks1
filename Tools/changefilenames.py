import os


def remove_strings(n, name):
    return name[n:]


directory = 'C:\\Users\\mans3851\\Desktop\\Processed e57'

for folders in os.listdir(directory):
    parent_dir = os.path.join(directory, folders)
    for file_name in os.listdir(parent_dir):
        new_name = remove_strings(4, file_name)
        abs_dir = os.path.join(parent_dir, file_name)
        abs_dir_new = os.path.join(parent_dir, new_name)
        os.rename(abs_dir, abs_dir_new)
