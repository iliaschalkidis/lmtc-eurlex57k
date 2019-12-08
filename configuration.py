import json
import os
import logging
from time import gmtime, strftime

from data import LOGGING_DIR

LOGGER = logging.getLogger(__name__)

parameters = {}


class ParameterStore(type):
    def __getitem__(cls, key: str):
        global parameters
        return parameters[key]

    def __setitem__(cls, key, value):
        global parameters
        parameters[key] = value


class Configuration(object, metaclass=ParameterStore):

    @staticmethod
    def configure():
        global parameters

        with open(os.path.join('lmtc_configuration.json')) as config_file:
            parameters = json.load(config_file)

        # Setup Logging
        log_name = '{}_{}_{}_{}'.format(Configuration['task']['dataset'].upper(),
                                        'HIERARCHICAL' if Configuration['sampling']['hierarchical'] else 'FLAT',
                                        Configuration['model']['architecture'].upper(), strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

        parameters['task']['log_name'] = log_name
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(LOGGING_DIR, log_name + '.txt'),
                            filemode='a')

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    @classmethod
    def __getitem__(cls, item: str):
        global parameters
        return parameters[item]
