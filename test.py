import os
os.system('gdown "1-20p3fet95niutbpkV636INRvS8EWJFF" --output dataset.zip')
os.system('unzip -q -o dataset.zip')
os.system('rm -f -r dataset.zip')


from app.models.classification.timm.train import Trainer
trainer = Trainer()
trainer.train(
    data_set='/Users/letienloc/Desktop/colab_test/timm_test/drive/Shareddrives/Viettel_AI_Intern/Task/Anti-Spoofing-Classification/dataset/curated_dataset_v1',
    num_epochs=15,
)

