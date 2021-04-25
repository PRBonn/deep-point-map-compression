#!/usr/bin/env python3

from depoco.trainer import DepocoNetTrainer
from ruamel import yaml
import argparse
import time
import depoco.utils.point_cloud_utils as pcu
import os

if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./evaluate.py")
    parser.add_argument(
        '--config_cfg', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='configitecture yaml cfg file. See /config/config for sample. No default!',
    )
    parser.add_argument(
        '--file_ext', '-fe',
        type=str,
        required=False,
        default='',
        help='Extends the output file name by the given string',
    )
    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
    config = yaml.safe_load(open(FLAGS.config_cfg, 'r'))
    print('loaded yaml flags')
    print('config:', FLAGS.config_cfg)
    trainer = DepocoNetTrainer(config)
    print('initialized  trainer')
    # trainer.train()
    ts = time.time()
    test_dict = trainer.test(
        best=True)
    print('evaluation time:', time.time()-ts)
    # trainer.validate(load_model=True)
    print('rec_err', test_dict['mapwise_reconstruction_error'][0:10])
    print('memory', test_dict['memory'][0:10])

    if not os.path.exists(config['evaluation']['out_dir']):
        os.makedirs(config['evaluation']['out_dir'])

    file = config['evaluation']['out_dir'] + \
        trainer.experiment_id+FLAGS.file_ext+".pkl"
    pcu.save_obj(test_dict, file)
