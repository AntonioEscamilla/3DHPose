from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import OrderedDict

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model


class FeatureExtractor(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=True,
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        num_params, flops = compute_model_complexity(
            model, (1, 3, image_size[0], image_size[1])
        )

        if verbose:
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)


        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def pairwise_affinity(self,query_features, gallery_features, query=None, gallery=None):
        # import pdb; pdb.set_trace()
        x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        # x si y vor fi tensori cu prima dimensiune sizeul si a doua sa se potriveasca deci tot sizeul
        x = x.view(m, -1)
        y = y.view(n, -1)
        # cred ca aici se face distanta euclidiana dintre descriptorii de bounding boxesu-uri
        # sunt foarte prost, variabila chiar se numeste dist!!!
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        normalized_affinity = - (dist - dist.mean()) / dist.std()
        affinity = torch.sigmoid(normalized_affinity * torch.tensor(5.))  # x5 to match 1->1
        # pro = x @ y.t ()
        # norms = x.norm ( dim=1 ).unsqueeze ( 1 ) @ y.norm ( dim=1 ).unsqueeze ( 0 )
        # affinity = (pro / norms + 1) / 2  # map from (-1, 1) to (0, 1)
        # affinity = torch.sigmoid ( pro / norms ) #  map to (0, 1)
        return affinity
    def extract_cnn_feature(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features

    def extract_features(self,data_batch):

        features = OrderedDict()
        imgs, fnames, pids, cam_id = data_batch
        # data_time.update ( time.time () - end )
        outputs = self.extract_cnn_feature(imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            # if isinstance ( fname, list ):
            #     fname = fname[0]
            features[fname] = output
        return features

    def get_affinity(self, query_batch, output_feature='pool5', rerank=False, use_gpu=True):
        """
        Will compute matrix with itself
        :param query_batch: An iterable object with (this_imgs, fnames, pids, _)
        :param output_feature:'pool5'
        :param rerank: boolean
        :param use_gpu: boolean
        :return:
        """
        query = list ( zip ( *query_batch[1:] ) )
        # Adapt to use dataloader
        # query = list ( map ( lambda x: [i[0] for i in x], query ) )
        #efectiv trece prin retea inputurile si ia de la pool5 vectorii si ii pune intr-un dictionar ordonat de features
        query_features = self.extract_features(query_batch)
        #if rerank:
         #   affinity = reranking ( query_features, query_features.copy (), query, query )
        #else:
        affinity = self.pairwise_affinity ( query_features, query_features.copy (), query, query )
        return affinity
