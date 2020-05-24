from skimage.transform import resize


def box2image(image, box, image_size=(28, 28)):
    """
    Extracts objects from image bounded by box
    :param image:
    :param box:
    :param image_size: return image size
    :return: object bounded by box on original image
    """
    return resize(image[box[1]:box[3], box[0]:box[2]], image_size)


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output
