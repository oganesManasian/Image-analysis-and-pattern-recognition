import argparse

from main import main
from perturb_video import rotate_and_pert_brightness
from video import read_video
from argparse import Namespace


def parse_arguments():
    parser = argparse.ArgumentParser("Video perturbation")
    parser.add_argument('--original', default="input_videos/robot_parcours_1.avi", type=str,
                        help='Path to the original input video')
    parser.add_argument('--input', default="perturb_videos/input/", type=str,
                        help='Path to the folder with saved input perturbated videos')
    parser.add_argument('--output', default="perturb_videos/output/", type=str,
                        help='Path to the folder with saved output perturbated videos')
    parser.add_argument('--n', default=10, type=int,
                        help='Number of perturb videos to generate')
    args = parser.parse_args()
    return args

def check_main(args):
    
    frames = read_video(args.original)
    expressions = []
    
    for i in range(1, args.n+1):
    
        _ = rotate_and_pert_brightness(frames, args.input + f'perturb_{i}.avi')
    
        print(f'Process {i} generated video\n')
        
        args_to_main = Namespace(input=args.input + f'perturb_{i}.avi', output=args.output + f'perturb_{i}.avi')

        main(args_to_main)
        
        print('\n\n\n')
    
    print('Finished processing')
    
        
if __name__ == "__main__":
    args = parse_arguments()
    check_main(args)