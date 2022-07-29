import os
from pathlib import Path
import sys
import time
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter

torch.backends.cudnn.benchmark = True


class Inference:
	def __init__(self,
				 checkpoint_path,
				 model_name: str = "mixnet_s",
				 num_classes: int = 2,
				 no_test_pool: bool = True):
		self.num_classes = num_classes
		self.no_test_pool = no_test_pool
		self.k = 1  # Always take top 1@
  
		model = create_model(
			model_name,
			num_classes=num_classes,
			in_chans=3,
			checkpoint_path=checkpoint_path,
		)
		self.config = resolve_data_config(self.__init_config(), model=model)
		self.model, self.test_time_pool = (model, False) if no_test_pool else apply_test_time_pool(model, self.config)

	def run(self,
			source: str,
			num_gpu: int = 1,
			batch_size: int = 32,
			workers: int = 4):

		# TODO: Run by CPU
		if torch.cuda.is_available():
			if num_gpu > 1:
				self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpu))).cuda()
			else:
				self.model = self.model.cuda()

		loader = create_loader(
			ImageDataset(source),
			input_size=self.config['input_size'],
			no_aug=False,
			batch_size=batch_size,
			use_prefetcher=True,
			interpolation=self.config['interpolation'],
			mean=self.config['mean'],
			std=self.config['std'],
			num_workers=workers,
			crop_pct=1.0 if self.test_time_pool else self.config['crop_pct'],
		)

		self.model.eval()

		batch_time = AverageMeter()
		end = time.time()
		topk_ids = []
		with torch.no_grad():
			for _, (input, _) in enumerate(loader):
				input = input.cuda()
				labels = self.model(input)
				topk = labels.topk(self.k)[1]
				topk_ids.append(topk.cpu().numpy())

				# measure elapsed time
				batch_time.update(time.time() - end)
				end = time.time()
		return topk_ids

	@staticmethod
	def __init_config():
		return {
			'input_size': (3, 400, 400),
			'mean': IMAGENET_DEFAULT_MEAN,
			'std': IMAGENET_DEFAULT_STD,
			'interpolation': 'bilinear',
			'crop_pct': 0.4,
		}
