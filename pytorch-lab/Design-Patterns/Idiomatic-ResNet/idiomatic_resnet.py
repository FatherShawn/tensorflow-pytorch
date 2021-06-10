import typing
import torch
import torch.nn as nn
import itertools

# This model uses 224 x 224 color images.

class ResStem(nn.Module):

    def __init__(self):
        # Call the parent constructor.
        super().__init__()

        # Define the layers.
        # First the convolutional blocks
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, inputs):
        # This defines a forward pass for this forward feed network.
        return self.layers(inputs)

class ResGroup(typing.TypedDict):
    """
    A data structure for the Residual Groups.
    """
    n_filters: int
    projection_block: nn.Sequential
    identity_blocks: tuple[nn.Sequential, ...]

class ResLearner(nn.Module):

    def __init__(self, group_parameters: list[tuple[int, int], ...]):
        """
        Parameters
        ----------
        group_parameters: list of tuple
            tuple: The number of filters, the number of blocks.

            To implement common ResNet configurations, pass the following lists:
            ResNet50:  [ (64, 3), (128, 4), (256, 6),  (512, 3) ]
            ResNet101: [ (64, 3), (128, 4), (256, 23), (512, 3) ]
            ResNet152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]
        """
        # Call the parent constructor.
        super().__init__()

        # Define the groups.
        self.__groups: list[ResGroup] = []

        # First Residual Block Group (not strided)
        n_filters, n_blocks = group_parameters.popleft()
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
        for group in itertools.islice(self.__groups, 1):
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
        x = group['projection_block'](x) + self.__add_projection_shortcut(x, group['n_filters'], stride)
        for block in group['identity_blocks']:
            x = block(x)
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
        identities = []
        for _ in range(n_blocks):
            identities.append(self.__standard_residual_block(n_filters))
        return ResGroup(
            n_filters=n_filters,
            projection_block=self.__projection_residual_block(n_filters, strides),
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
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(3, n_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        # Bottleneck layer
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(3, n_filters, kernel_size=(3, 3), stride=(1, 1))
        ]
        # Dimensionality restoration - increase the number of filters by 4X
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(3, 4 * n_filters, kernel_size=(1, 1), stride=(1, 1))
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
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(3, n_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        # Bottleneck layer
        # Feature pooling when strides=(2, 2)
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(3, n_filters, kernel_size=(3, 3), stride=(strides, strides), padding=(1, 1), padding_mode='replicate')
        ]
        # Dimensionality restoration - increase the number of filters by 4X
        layers += [
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(3, 4 * n_filters, kernel_size=(1, 1), stride=(1, 1))
        ]
        return nn.Sequential(*layers)

    def __add_projection_shortcut(self, inputs: torch.Tensor, n_filters: int, strides: int = 2) -> torch.Tensor:
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
        torch.Tensor
            The layers for a Sequential model
        """
        # Increase filters by 4X to match shape when added to output of block
        model = nn.Sequential(
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(3, 4 * n_filters, kernel_size=(1, 1), stride=(strides, strides)),
        )
        shortcut = model(inputs)
        return inputs + shortcut