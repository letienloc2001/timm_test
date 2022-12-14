from app.models.classification.timm.inference import Inference
from app.models.classification.timm.train import Trainer
import yaml


class Classification:
	""" Classification model """

	def __init__(self,
				 checkpoint_path: str,
				 model_name: str = "mixnet_s",
				 num_classes: int = 2,
				 pretrained: bool = False):
		self.model_name = model_name
		self.num_classes = num_classes
		self.checkpoint_path = checkpoint_path
		self.model = Inference(
			checkpoint_path,
			model_name=model_name,
			num_classes=num_classes,
			no_test_pool=True,
		)
		self.trainer = Trainer(model_name=model_name,
											 checkpoint_path=checkpoint_path,
											 num_classes=num_classes,
											 pretrained=pretrained)

	def infer(self, source):
		return self.model.run(source=source, batch_size=8)

	def train(self,
           	  data_set: str, 
              configuration: str):
		"""
			Train model and save its checkpoint

		Args:
			data_set (str): path to dataset
			configuration (str): path to config.yaml

		Returns:
			the best checkpoint path
		"""
		# TODO: get configuaration from DB
		with open(configuration, 'r') as f:
			cfg = yaml.safe_load(f)

		best_model_path = self.trainer.train(
			data_set=data_set,
			num_epochs=cfg['epochs'],
			learning_rate=cfg['lr'],
			weight_decay=cfg['weight_decay'],
			momentum=cfg['momentum'],
			batch_size=cfg['batch_size'],
			best_model_path=cfg['outpath'],
			reset_training=cfg['reset_training']
		)

		# TODO: Validate performance before assigning to the instance
		self.model = Inference(
			best_model_path,
			model_name=self.model_name,
			num_classes=self.num_classes,
			no_test_pool=True,
		)

		return best_model_path
