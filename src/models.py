import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models.py')

def get_model(args):
    logger.info(f'Fetching model. args.model {args.model} | args.model_size: {args.model_size}')
    # TODO
