import os
path = os.path.split(os.path.realpath(__file__))[0]
root_directory =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(path)
print(root_directory)