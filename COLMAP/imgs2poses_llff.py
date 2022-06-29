from llff.poses.pose_utils import gen_poses
import sys
import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('--match_type', type=str, 
					default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
parser.add_argument('--scenedir', type=str,
                    help='input scene directory')
parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
parser.add_argument('--test_path', help='directory where to save the tests')
args = parser.parse_args()

if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
	print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
	sys.exit()

if __name__=='__main__':
    # source ~/.bashrc
    # python3 imgs2poses_llff.py -c ./configs/llff.conf
    gen_poses(args.scenedir, args.match_type, test_path=args.test_path)