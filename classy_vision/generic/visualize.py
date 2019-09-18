#!/usr/bin/env python3

import math

import numpy as np
import torch
import torch.nn.modules as nn
from classy_vision.generic.util import is_pos_int
from PIL import Image


try:
    import visdom
    from tensorboardX import SummaryWriter
except ImportError:
    pass

# define loss types:
vis = []  # using a list makes this work as an upvalue
UNSUPPORTED_LOSSES = (
    nn.CosineEmbeddingLoss,
    nn.PoissonNLLLoss,
    nn.KLDivLoss,
    nn.TripletMarginLoss,
)
REGRESSION_LOSSES = (nn.L1Loss, nn.SmoothL1Loss, nn.MSELoss)


# connection to visdom:
def visdom_connect(server=None, port=None):
    """Connects to a visdom server if not currently connected."""
    if not visdom_connected():
        vis.append(visdom.Visdom(server=server, port=port))


# check if we are connected to visdom:
def visdom_connected():
    """Returns True if the client is connected to a visdom server."""
    return (
        len(vis) > 0
        and hasattr(vis[-1], "check_connection")
        and vis[-1].check_connection()
    )


# function that plots learning curve:
def plot_learning_curves(curves, visdom_server=None, env=None, win=None, title=""):
    """Plots the specified dict of learning curves in visdom. Optionally, the
    environment, window handle, and title for the visdom plot can be specified.
    """

    if visdom_server is None and visdom_connected():
        visdom_server = vis[-1]

    # return if we are not connected to visdom server:
    if not visdom_server or not visdom_server.check_connection():
        print("WARNING: Not connected to visdom. Skipping plotting.")
        return

    # assertions:
    assert type(curves) == dict
    assert all(type(curve) == list for _, curve in curves.items())

    # remove batch time curves:
    _curves = {k: curves[k] for k in curves.keys() if "batch time" not in k}

    # show plot:
    X = torch.stack([torch.FloatTensor(curve) for _, curve in _curves.items()], dim=1)
    Y = torch.arange(0, X.size(0))
    Y = Y.view(Y.numel(), 1).expand(Y.numel(), X.size(1))
    opts = {"title": title, "legend": list(_curves.keys()), "xlabel": "Epochs"}
    return visdom_server.line(X, Y, env=env, win=win, opts=opts)


# function that plots loss functions:
def plot_losses(losses, visdom_server=None, env=None, win=None, title=""):
    """Constructs a plot of specified losses as function of y * f(x). The losses
    are a list of nn.Module losses. Optionally, the environment, window handle,
    and title for the visdom plot can be specified.
    """

    if visdom_server is None and visdom_connected():
        visdom_server = vis[-1]

    # return if we are not connected to visdom server:
    if not visdom_server or not visdom_server.check_connection():
        print("WARNING: Not connected to visdom. Skipping plotting.")
        return

    # assertions:
    if isinstance(losses, nn.Module):
        losses = [losses]
    assert type(losses) == list
    assert all(isinstance(loss, nn.Module) for loss in losses)
    if any(isinstance(loss, UNSUPPORTED_LOSSES) for loss in losses):
        raise NotImplementedError("loss function not supported")

    # loop over all loss functions:
    for idx, loss in enumerate(losses):

        # construct scores and targets:
        score = torch.arange(-5.0, 5.0, 0.005)
        if idx == 0:
            loss_val = torch.FloatTensor(score.size(0), len(losses))
        if isinstance(loss, REGRESSION_LOSSES):
            target = torch.FloatTensor(score.size()).fill_(0.0)
        else:
            target = torch.LongTensor(score.size()).fill_(1)

        # compute loss values:
        for n in range(0, score.nelement()):
            loss_val[n][idx] = loss(
                score.narrow(0, n, 1), target.narrow(0, n, 1)
            ).item()

    # show plot:
    title = str(loss) if title == "" else title
    legend = [str(loss) for loss in losses]
    opts = {"title": title, "xlabel": "Score", "ylabel": "Loss", "legend": legend}
    win = visdom_server.line(loss_val, score, env=env, win=win, opts=opts)
    return win


def plot_model(model, size=(3, 224, 224), writer=None, folder="", train=True):
    """Visualizes a model in TensorBoard.

    The TensorBoard writer can be either specified directly via `writer` or can
    be specified via a `folder`.

    The model can be run in training or evaluation model via the `train` argument.

    Example usage on devserver:
     - Install TensorBoard using: `sudo feature install tensorboard`
     - Start TensorBoard using: `tensorboard --port=8098 --logdir <folder>`
    """

    assert (
        writer is not None or folder != ""
    ), "must specify SummaryWriter or folder to create SummaryWriter in"
    if len(size) == 3:
        size = (1,) + size
    assert len(size) == 4, "size must be tuple of length 4 (NCHW)"

    input = torch.zeros(size)
    if next(model.parameters()).is_cuda:
        input = input.cuda()
    if writer is None:
        writer = SummaryWriter(log_dir=folder, comment="Model graph")
    with writer:
        orig_train = model.training
        model.train(train)  # visualize model in desired mode
        writer.add_graph(model, input_to_model=(input,))
        model.train(orig_train)


# function that produces an image map:
def image_map(
    mapcoord, dataset, mapsize=5000, imsize=32, unnormalize=None, snap_to_grid=False
):
    """Constructs a 2D map of images.

    The 2D coordinates for each of the images are specified in `mapcoord`, the
    corresponding images are in `dataset`. Optional arguments set the size of
    the map images, the size of the images themselves, the unnormalization
    transform, and whether or not to snap images to a grid.
    """

    # assertions:
    if type(mapcoord) == np.ndarray:
        mapcoord = torch.from_numpy(mapcoord)
    assert torch.is_tensor(mapcoord)
    if isinstance(dataset, torch.utils.data.dataloader.DataLoader):
        dataset = dataset.dataset
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert is_pos_int(mapsize)
    assert is_pos_int(imsize)
    if unnormalize is not None:
        assert callable(unnormalize)

    # initialize some variables:
    import torchvision.transforms.functional as F

    background = 255
    mapim = torch.ByteTensor(3, mapsize, mapsize).fill_(background)

    # normalize map coordinates:
    mapc = mapcoord.add(-mapcoord.min())
    mapc.div_(mapc.max())

    # loop over images:
    for idx in range(len(dataset)):

        # compute grid location:
        if snap_to_grid:
            y = 1 + int(math.floor(mapc[idx][0] * (mapsize - imsize - 2)))
            x = 1 + int(math.floor(mapc[idx][1] * (mapsize - imsize - 2)))
        else:
            y = 1 + int(math.floor(mapc[idx][0] * (math.floor(mapsize - imsize) - 2)))
            x = 1 + int(math.floor(mapc[idx][1] * (math.floor(mapsize - imsize) - 2)))

        # check whether we can overwrite this location:
        overwrite = not snap_to_grid
        if not overwrite:
            segment = mapim.narrow(1, y, imsize).narrow(2, x, imsize)
            overwrite = segment.eq(background).all()

        # draw image:
        if overwrite:

            # load, unnormalize, and resize image:
            image = dataset[idx][0]
            if unnormalize is not None:
                image = unnormalize(image)
            resized_im = F.to_tensor(
                F.resize(F.to_pil_image(image), imsize, Image.BILINEAR)
            )

            # place image:
            segment = mapim.narrow(1, y, imsize).narrow(2, x, imsize)
            segment.copy_(resized_im.mul_(255.0).byte())

    # return map:
    return mapim
