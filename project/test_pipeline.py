from main import main
from perturb_video import generate_new_video


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


NUM_TESTS = 10

TARGET = "3/2+7*2="
results = []

for i in range(NUM_TESTS):
    src_file_path = generate_new_video()
    args = Namespace(input=src_file_path,
                     output="videos/output.avi")
    try:
        predicted = main(args)
    except SyntaxError:
        predicted = None
    res = "passed" if predicted == TARGET else "NOT passed"
    print(f"Test {i} ({src_file_path}), predicted: {predicted}. Test is {res}")
    results.append([src_file_path, res])

print("Evaluation results")
for i, (src_file_path, res) in enumerate(results):
    print(f"Test {i} ({src_file_path}) test is {res}")
