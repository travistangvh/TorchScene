import logging
import hydra
from omegaconf import DictConfig
import torch
from torchvision import transforms
from PIL import Image
from train import initialize_model
import os

class torchscene():

    # @hydra.main(version_base=None, config_path="../conf", config_name="infer")
    def __init__(self, ):
        """a simple model inference script for scene recognizing(places365 challenge).
            http://places2.csail.mit.edu/download.html
        """
        hydra.initialize(version_base=None, config_path="../conf")
        self.cfgs = hydra.compose(config_name="infer")
        self.device = torch.device(self.cfgs.device) if torch.cuda.is_available() else torch.device('cpu')
        self.categories_map_dir = self.cfgs.categories_map_dir
        self.logger = logging.getLogger('model inference')
        self.logger.info("infer on device:{}".format(self.device))
        self.model, self.input_size = initialize_model(self.cfgs.arch, num_classes=365, feature_extract=False, use_pretrained=False)
        # torch.save(self.model.state_dict(), self.cfgs.weight_path)
        self.logger.info("loading model weights from {}".format(self.cfgs.weight_path))
        self.model.load_state_dict(torch.load(self.cfgs.weight_path))
        self.logger.info("loading successfully.")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.data_transform = transforms.Compose([
            transforms.Resize(
                self.input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def pred(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            self.logger.info("input image is loaded from {}".format(img_path))
            input = self.data_transform(img).unsqueeze(0).to(self.device)
            outputs = torch.softmax(self.model(input).to('cpu'), dim=-1)
            predict = int(torch.argmax(outputs))
            with open(self.categories_map_dir, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if predict == int(line.split(' ')[-1]):
                        self.logger.info("prediction of {} is {}".format(img_path, line.split(' ')[0].split('/')[-1]))
                        break
            return outputs
        except:
            self.logger.info(f'{img_path} failed.')
            return torch.Tensor([0]).repeat([365])[None, :]


# if __name__ == "__main__":
#     main()





