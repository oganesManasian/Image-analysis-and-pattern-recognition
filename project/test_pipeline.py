import os

from main import main
from perturb_video import generate_new_video


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


TEST_ON_AVAILABLE_VIDEOS = True
NUM_VIDEOS_TO_GENERATE = 10

TARGET = "3/2+7*2="
results = []
if TEST_ON_AVAILABLE_VIDEOS:
    files = [os.path.join("videos", filename) for filename in os.listdir("videos") if filename != "output.avi"]
else:
    files = [generate_new_video() for i in range(NUM_VIDEOS_TO_GENERATE)]

for i, src_file_path in enumerate(files):
    args = Namespace(input=src_file_path,
                     output="videos/output.avi")
    try:
        predicted = main(args)
    except SyntaxError:
        predicted = None
    res = "passed" if predicted == TARGET else f"NOT passed: target: {TARGET}, predicted: {predicted}"
    print(f"Test {i} ({src_file_path}), predicted: {predicted}. Test is {res}")
    results.append([src_file_path, res])

print("Evaluation results")
for i, (src_file_path, res) in enumerate(results):
    print(f"Test {i} ({src_file_path}) test is {res}")
