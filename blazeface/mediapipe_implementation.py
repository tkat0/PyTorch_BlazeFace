from torch import nn

from blazeface import BlazeBlock


def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


class MediaPipeBlazeFace(nn.Module):
    """Constructs a BlazeFace model of the MediaPipe implementation

    the original implementation
    https://github.com/google/mediapipe/tree/master/mediapipe/models#blazeface-face-detection-model
    """

    def __init__(self):
        super(MediaPipeBlazeFace, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24, kernel_size=3),
            BlazeBlock(24, 28, kernel_size=3),
            BlazeBlock(28, 32, kernel_size=3, stride=2),
            BlazeBlock(32, 36, kernel_size=3),
            BlazeBlock(36, 42, kernel_size=3),
            BlazeBlock(42, 48, kernel_size=3, stride=2),
            BlazeBlock(48, 56, kernel_size=3),
            BlazeBlock(56, 64, kernel_size=3),
            BlazeBlock(64, 72, kernel_size=3),
            BlazeBlock(72, 80, kernel_size=3),
            BlazeBlock(80, 88, kernel_size=3),
            BlazeBlock(88, 96, kernel_size=3, stride=2),
            BlazeBlock(96, 96, kernel_size=3),
            BlazeBlock(96, 96, kernel_size=3),
            BlazeBlock(96, 96, kernel_size=3),
            BlazeBlock(96, 96, kernel_size=3),
        )

        self.apply(initialize)

    def forward(self, x):
        h = self.features(x)
        return h
