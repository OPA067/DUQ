from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from torch.utils.data import DataLoader

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        else:
            raise NotImplementedError
