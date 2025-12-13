import torch.nn as nn

class Refiner(nn.Module):
    def __init__(self, input_size, output_size, d_model=32, num_layers=3, kernel_size=3):
        super(Refiner, self).__init__()

        in_channels = input_size
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2

            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(d_model))
            in_channels = d_model

        layers.append(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=output_size,
                kernel_size=1,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)