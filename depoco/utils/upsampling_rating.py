import numpy as np
import argparse

def getScalingFactors(nr_in, nr_out, nr_layer):
    sf = (nr_out/nr_in)**(1/nr_layer)
    factors = nr_layer*[int(sf)]
    factors[-1]=round(nr_out/(np.prod(factors[:-1])*nr_in))

    sampling_factor = np.prod(factors)
    print(f'factors {factors}, upsampling rate {sampling_factor}, nr output points {sampling_factor*nr_in}')



if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./upsample_rating.py")
    # parser.add_argument(
    #     '--dataset', '-d',
    #     type=str,
    #     required=False,
    #     default="/mnt/91d100fa-d283-4eeb-b68c-e2b4b199d2de/wiesmann/data/data_kitti/dataset/" ,
    #     help='Dataset to train with. No Default',
    # )
    parser.add_argument(
        '--input', '-i',
        type=int,
        required=True,
        default=120,
        help='Nr input points',
    )
    parser.add_argument(
        '--output', '-o',
        type=int,
        required=True,
        default=100000,
        help='Nr output points',
    )
    parser.add_argument(
        '--layer', '-l',
        type=int,
        required=False,
        default=4,
        help='Nr layer',
    )
    FLAGS, unparsed = parser.parse_known_args()

    getScalingFactors(FLAGS.input,FLAGS.output,FLAGS.layer)