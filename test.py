# import logging
# import time
# import enlighten
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()
# # Setup progress bar
# manager = enlighten.get_manager()
# pbar = manager.counter(total=100, desc='Train', unit='batches')

# for i in range(1, 101):
#     # logger.info("Processing step %s" % i)
#     time.sleep(.2)
#     pbar.update()
# import logging
# import time
# import colorlog
# from tqdm import tqdm

# class TqdmHandler(logging.StreamHandler):
#     def __init__(self):
#         logging.StreamHandler.__init__(self)

#     def emit(self, record):
#         msg = self.format(record)
#         tqdm.write(msg)

# if __name__ == "__main__":
#     for x in tqdm(range(100)):
#         logger = colorlog.getLogger("MYAPP")
#         logger.setLevel(logging.DEBUG)
#         handler = TqdmHandler()
#         logger.addHandler(handler)
#         time.sleep(.5)
# import tqdm
# import time
# outer = tqdm.tqdm(total=100, desc='Epoch', position=0)
# for ii in range(100):
#     outer.update(1)
#     time.sleep(0.5)

# import logging
# # create console handler
# ch = logging.StreamHandler()
# # create formatter
# formatter = logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(message)s')
# # add formatter to console handler
# ch.setFormatter(formatter)
# # add console handler to logger
# logger.addHandler(ch)

import logging
import time
logger = logging.getLogger('njns')

for i in range(100):
    logger.info(f'{i}')
    time.sleep(1)