import typing
import torch
import torch.nn as nn
import torch.nn.functional as activation
import itertools

# This model uses 256 x 256 color images.

##################################################################################
# The Idomatic architecture pattern is three-fold:
#
# Stem: This is the entry point.  It does course feature extraction and
#       adjusts the shape of the data for the Learner component.
# Learner: This is where the convolutions are for learning the features
#          of the images.
# Classifier: Maps the features to one or more image categories. Ex: dog, cat
#
# This pattern is coded below with a class for each stage of the architecture.
# The ResNet itself then is a simple object composed of an instance of each
# stage.
##################################################################################

class ResStem(nn.Module):

    def __init__(self) -> None:
        # Call the parent constructor.
        super().__init__()

        # Define the layers.
        # First the convolutional blocks
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, inputs):
        # This defines a forward pass for this forward feed network.
        return self.layers(inputs)

class ResGroup(typing.TypedDict):
    """
    A data structure for the Residual Groups.
    """
    projection_block: nn.Sequential
    shortcut: nn.Sequential
    identity_blocks: tuple[nn.Sequential, ...]

class ResLearner(nn.Module):

    def __init__(self, group_parameters: list[tuple[int, int], ...], incoming_filters: int = 64) -> None:
        """
        Parameters
        ----------
        group_parameters: list of tuple
            tuple: The number of filters, the number of blocks.

            To implement common ResNet configurations, pass the following lists:
            ResNet50:  [ (64, 3), (128, 4), (256, 6),  (512, 3) ]
            ResNet101: [ (64, 3), (128, 4), (256, 23), (512, 3) ]
            ResNet152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]
        incoming_filters: int
          The number of filters set in the output of the Stem component. Typically set to 64.
        """
        # Call the parent constructor.
        super().__init__()

        # Track the number of output channels in the prior block for use as input channels in the next block.
        self.__trailing_filters = incoming_filters
        # Define the groups.
        self.__groups: list[ResGroup] = []

        # First Residual Block Group (not strided)
        n_filters, n_blocks = group_parameters.pop(0)
        self.__groups.append(self.__build_group(n_filters, n_blocks, 1))

        # Remaining Residual Block Groups (strided with default of 2)
        for n_filters, n_blocks in group_parameters:
            self.__groups.append(self.__build_group(n_filters, n_blocks))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Moves an input value through the neural networks.

        Parameters
        ----------
        x: torch.Tensor
          The input

        Returns
        -------
        torch.Tensor
          The output
        """
        # The first group is not strided (stride = 1).
        group = self.__groups[0]
        x = self.__forward_group(x, group, 1)

        # Now feed through the remaining groups.
        for group in itertools.islice(self.__groups, 1, None):
            x = self.__forward_group(x, group)

        return x

    def __forward_group(self, x: torch.Tensor, group: ResGroup, stride:int = 2) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.Tensor
          The input
        group: ResGroup
          The group through which x is fed.
        stride: int
          The size of the stride used by the projection shortcut.

        Returns
        -------
        torch.Tensor
          The output from the layers in the group.
        """
        shortcut = group['shortcut'](x)
        x = group['projection_block'](x)
        x = x + shortcut
        for block in group['identity_blocks']:
            shortcut = x
            x = block(x)
            x = x + shortcut
        return x

    def __build_group(self, n_filters, n_blocks, strides=2) -> ResGroup:
        """
        Builds a dictionary of blocks to model a residual group for the Idiomatic ResNet architecture.

        Parameters
        ----------

        n_filters: integer
            The number of filters.
        n_blocks: integer
            The number of blocks in the group.
        strides: integer
            The stride to use for the Conv layers in the group

        Returns
        -------
        ResGroup
        """
        shortcut = self.__get_projection_shortcut(n_filters, strides)
        projection = self.__projection_residual_block(n_filters, strides)
        identities = []
        for _ in range(n_blocks):
            identities.append(self.__standard_residual_block(n_filters))
        return ResGroup(
            projection_block=projection,
            shortcut=shortcut,
            identity_blocks=tuple(identities)
        )

    def __standard_residual_block(self, n_filters: int) -> nn.Sequential:
        """
        Assemble a Sequential for a Bottleneck Residual Block
        of Convolutions for use with a Identity Shortcut

        Parameters
        ----------
        n_filters: int
            The number of filters used in each layer.

        Returns
        -------
        nn.Sequential
            A Sequential model object.
        """
        layers = []

        ## Construct the 1x1, 3x3, 1x1 convolution block ##

        # Dimensionality reduction
        layers += [
            nn.BatchNorm2d(self.__trailing_filters),
            nn.ReLU(),
            nn.Conv2d(self.__trailing_filters, n_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        # Bottleneck layer
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='replicate')
        ]
        # Dimensionality restoration - increase the number of filters by 4X
        self.__trailing_filters = 4 * n_filters
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, self.__trailing_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        return nn.Sequential(*layers)

    def __projection_residual_block(self, n_filters: int , strides: int =2) -> nn.Sequential:
        """
        Assemble a Sequential for a Bottleneck Residual Block
        of Convolutions for use with a Projection Shortcut

          Padding values
          s: stride
          k: kernel size
                |   s   |
                | 1 | 2 |
        +---+---+---+---+
        |   | 1 | 0 | 0 |
        | k |---+---+---+
        |   | 3 | 1 | 1 |

        Parameters
        ----------
        n_filters: int
            The number of filters used in each layer.
        strides: int
            The strides used in the layers of the block.

        Returns
        -------
        nn.Sequential
            A Sequential model object.
        """

        layers = []

        ## Construct the 1x1, 3x3, 1x1 convolution block ##

        # Dimensionality reduction
        layers += [
            nn.BatchNorm2d(self.__trailing_filters),
            nn.ReLU(),
            nn.Conv2d(self.__trailing_filters, n_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        # Bottleneck layer
        # Feature pooling when strides=(2, 2)
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(strides, strides), padding=(1, 1), padding_mode='replicate')
        ]
        # Dimensionality restoration - increase the number of filters by 4X
        self.__trailing_filters = 4 * n_filters
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, self.__trailing_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        return nn.Sequential(*layers)

    def __get_projection_shortcut(self, n_filters: int, strides: int = 2) -> nn.Sequential:
        """
        Process a projection shortcut.


        Parameters
        ----------
        inputs: tensor
            An input tensor.
        n_filters: integer
            The number of filters in the block.
        strides: integer
            The stride to use for the Conv layer in the block

        Returns
        -------
        nn.Sequential
            The Sequential model for a shortcut block.
        """
        return nn.Sequential(
            # Channels entering the projection block will be self.__trailing_filters.
            # Channels exiting the projection block will be 4 * n_filters so match the shape.
            nn.BatchNorm2d(self.__trailing_filters),
            nn.Conv2d(self.__trailing_filters, 4 * n_filters, kernel_size=(1, 1), stride=(strides, strides)),
        )

class ResClassifier(nn.Module):

    def __init__(self, n_filters: int, n_classes: int) -> None:
        """
        Constructor for the ResClassifier.

        Parameters
        ----------
        n_filters: int
            The number of incoming filters.  4 * the final n_filters for the learner blocks if paired with ResLearner.
        n_classes: int
            The number of classes into which images are sorted.
        """
        # Call the parent constructor.
        super().__init__()

        self.denseLayer = nn.Linear(n_filters, n_classes)
        self.globalAveragePool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.globalAveragePool(x)
        # Flatten for the dense layer
        x = torch.flatten(x, 1)
        return activation.softmax(self.denseLayer(x))

class ResNet(nn.Module):

    def __init__(self, stem: ResStem, learner: ResLearner, classifier: ResClassifier) -> None:
        # Call the parent constructor.
        super().__init__()

        self.stem = stem
        self.learner = learner
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.learner(x)
        return self.classifier(x)
