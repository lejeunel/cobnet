import configargparse
import argparse
from os.path import join as pjoin


def get_params(path='.'):
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[pjoin(path, 'default.yaml')])

    p.add('-v', help='verbose', action='store_true')

    p.add('--aug-n-angles', type=int)
    p.add('--epochs', type=int)
    p.add('--epochs-pre', type=int)
    p.add('--epochs-div-lr', type=int)
    p.add('--cp-period', type=int)
    p.add('--n-ims-test', type=int)
    p.add('--batch-size', type=int)
    p.add('--lr', type=float)
    p.add('--decay', type=float)
    p.add('--momentum', type=float)
    p.add('--cuda', default=False, action='store_true')

    return p
