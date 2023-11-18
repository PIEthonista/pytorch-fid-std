import argparse
from core.base_dataset import BaseDataset
from models.metric import inception_score


## to fix a import error for pytorch_fid
# def replace_line(file_name, line_num, text):
#     lines = open(file_name, 'r').readlines()
#     lines[line_num] = text
#     print(lines[line_num])
#     out = open(file_name, 'w')
#     out.writelines(lines)
#     print(lines[52])
#     out.close()
# replace_line('/content/pytorch-fid/src/pytorch_fid/fid_score.py', 51, 'from inception import InceptionV3\n')

# import sys
# sys.path.insert(1, '/content/pytorch-fid/src/pytorch_fid/')
# from fid_score import calculate_fid_given_paths

from src.pytorch_fid.fid_score import calculate_fid_given_paths
def fid_function(path=[], device='cuda', batch_size=50, num_workers=4, dims=2048):
    fid_value = calculate_fid_given_paths(path,
                                          batch_size,
                                          device,
                                          dims,
                                          num_workers)
    return fid_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')

    ''' parser configs '''
    args = parser.parse_args()

    fid_std_model = fid_function([args.src, args.dst])

    is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)
    
    print('  Std FID: {}'.format(fid_std_model))
    print('IS (mean): {}'.format(is_mean))
    print(' IS (std): {}'.format(is_std))
    
# results will be shown in slurm-JOBID.out file