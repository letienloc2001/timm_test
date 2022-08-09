# from app.services.classification import Classification

# cfg = dict(
#     epochs = 15,
#     lr = 0.0005,
#     weight_decay = 0.0009,
#     batch_size = 8,
#     momentum = 0.9,
#     outpath = 'checkpoint/mixnet_s-anti-spoofing-from-training.pth',
#     reset_training = False,
# )
# import yaml
# with open('test/anti-spoofing.yaml', 'w') as cfg_file:
#     yaml.dump(cfg, cfg_file, default_flow_style=False)

# anti_spoofing_trainer = Classification(checkpoint_path='')
# anti_spoofing_trainer.train('test/dataset', 'test/anti-spoofing.yaml')

# checkpoint_path = 'checkpoint/mixnet_s-anti-spoofing-from-training.pth' # 'checkpoint/mixnet_s-anti-spoofing.pth'
# anti_spoofing_model = Classification(checkpoint_path=checkpoint_path)

# fake_prediction = anti_spoofing_model.infer('test/dataset/test/fake')
# real_prediction = anti_spoofing_model.infer('test/dataset/test/real')

# print('🤥 FAKE PREDICTION: ', [e[0] for array in fake_prediction for e in array.tolist()])
# print('😘 REAL PREDICTION: ', [e[0] for array in real_prediction for e in array.tolist()])

import enlighten
import logging
logger = logging.getLogger()
pbar = enlighten.get_manager().counter(total=100, desc='Train     ', unit='it')

for i in range(100):
    pbar.update()
    
logger.info(f'Accuracy: 12.34%, Loss: 0.12345')


