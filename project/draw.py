import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw(image, trajectory=None, boxes=None, title=None, figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    if trajectory:
        X, Y = list(zip(*trajectory))
        plt.plot(X, Y, "g")
        plt.plot(X, Y, "go")

    if boxes:
        for i, box in enumerate(boxes):
            rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x=box[2], y=box[3], s=str(i))

    if title:
        plt.title(title)
    # plt.show()
