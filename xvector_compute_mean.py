from kaldi_python_io import ScriptReader
import argparse
import numpy as np


def script_reader(scp):
    reader = ScriptReader(scp, matrix=False)
    return reader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="take xvector.scp and compute global mean")
    parser.add_argument('-scp', type=str, help='path of xvector.scp')
    parser.add_argument('-out', type=str, help='output path for mean.npy')
    parser.add_argument('-spk2utt', type=str, default='', help='path to spk tu utt for speaker based mean compute')
    args = parser.parse_args()

    if not args.spk2utt:
        reader = script_reader(args.scp)
        sum = np.zeros((512,))
        num_done = 0
        for utt, xvec in reader:
            sum += xvec
            num_done += 1
        if num_done == 0:
            print("No ivectors read")
        else:
            mean = sum / num_done
            with open(args.out + '/mean.npy', 'wb') as f:
                np.save(f, mean)
