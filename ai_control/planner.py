# Samarah Uriarte & Prithika Ganesh

import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):

      # super(Planner, self).__init__()

      # Using same architecture as "Playing Atari with Deep Reinforcement Learning" (https://arxiv.org/pdf/1312.5602.pdf)
      # 128 x 96 -> size of input
      # Referencing https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/ for syntax

      # self.layer1 = torch.nn.Sequential(
      #   torch.nn.Conv2d(3,16,8,4,2),
      #   torch.nn.ReLU(),
      #   torch.nn.MaxPool2d(2,2)) # 64 x 48
      # self.layer2 = torch.nn.Sequential(
      #   torch.nn.Conv2d(16,32,4,2,2),
      #   torch.nn.ReLU(),
      #   torch.nn.MaxPool2d(2,2)) # 32 x 24
      # self.fc = torch.nn.Linear(32 * 24 * 32, 1)

        super().__init__()

        layers = []
        layers.append(torch.nn.Conv2d(3,16,8,4,2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(2,2))

        layers.append(torch.nn.Conv2d(16,32,4,2,2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(2,2))

        layers.append(torch.nn.Linear(32 * 24 * 32, 1))

        self._conv = torch.nn.Sequential(*layers)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        # x = self.layer1(img)
        # x = self.layer2(x)
        # x = x.reshape(x.size(0), -1) # 24576 x 1
        # x = self.drop_out(x)
        # x = self.fc(x)

        x = self._conv(img)
        
        #print(img.shape)
        #print(x.shape)
        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)