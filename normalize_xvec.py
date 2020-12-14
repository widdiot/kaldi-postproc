from kaldi_python_io import ScriptReader
import argparse
import numpy as np
from kaldiio import WriteHelper
import pickle


def script_reader(scp):
    reader = ScriptReader((scp), matrix=False)
    return reader


if __name__ == "__main__":
    """
    script subtracts mean, performs lda transform and then L@ normalizes the 
    xvector before saving it
    """

    parser = argparse.ArgumentParser(description="take xvector.scp and compute global mean")
    parser.add_argument('-scp', type=str, help='path of xvector.scp')
    parser.add_argument('-out', type=str, help='output path for normalized xvectors')
    parser.add_argument('-mean', type=str, default='', help='path to mean')
    parser.add_argument('-lda', type=str, default='', help='path to lda')
    args = parser.parse_args()

    with open(args.lda + "/lda.pkl", "rb") as input_file:
        lda = pickle.load(input_file)
    mean = np.load(args.mean+"/mean.npy")
    reader = script_reader(args.scp)
    num_done = 0
    with WriteHelper('ark,scp:' + args.out + '/xvector.ark,' + args.out + '/xvector.scp') as writer:
        for utt, xvec in reader:
            xvec_m = xvec - mean
            xvec_mLDA = lda.transform(np.expand_dims(xvec_m, axis=0), 150)
            xvec_mLDA = np.squeeze(xvec_mLDA)
            xvec_mLDAl2 = xvec_mLDA / (np.dot(xvec_mLDA, xvec_mLDA))
            writer(utt, xvec_mLDAl2)
