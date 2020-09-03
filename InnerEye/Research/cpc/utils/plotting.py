import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_encoding_space(encodings=None, precomputed_embeddings=None, color_by=None,
                        marker_by=None, title=None, out_path=None, scatter_kwargs=None, **tsne_kwargs):
    """
    Takes a [N, C] Tensor-like input and produces a 2D t-SNE plot over dim 1.
    Scatters may be colored and marked by arrays passed to color_by and marker_by.

    :param encodings:
    :param precomputed_embeddings:
    :param color_by:
    :param marker_by:
    :param title:
    :return:
    """
    if encodings is None and precomputed_embeddings is None:
        raise ValueError("Must pass either 'encodings' or 'precomputed_embeddings', got None and None")
    if precomputed_embeddings is None:
        encodings_2d = TSNE(**tsne_kwargs).fit_transform(encodings)
    else:
        encodings_2d = precomputed_embeddings
    assert encodings_2d.ndim == 2, "Expected encodings of ndim == 2 (shape [N, C]), " \
                                   "but got {} ()".format(encodings_2d.ndim, encodings_2d.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colors = color_by if color_by is not None else [0]*len(encodings_2d)
    markers = marker_by if marker_by is not None else [0]*len(encodings_2d)
    data = pd.DataFrame({"x": encodings_2d[:, 0],
                         "y": encodings_2d[:, 1],
                         "colors": colors,
                         "markers": markers})
    default_kwargs = {"s": 100, "alpha": 0.8, "picker": 2, "palette": "Set2"}
    default_kwargs.update(scatter_kwargs or {})
    sns.scatterplot(data=data, x="x", y="y", style="markers", hue="colors", ax=ax, **default_kwargs)
    if title:
        ax.set_title(title, size=26)
    ax.axis("off")
    if out_path is not None:
        fig.savefig(str(out_path), dpi=240)
        plt.close(fig)
    else:
        return fig, ax, encodings_2d


def onpick(event, encodings, images, segmentations, subjects, figures, axes, collection):
    """
    Event handler used by interactive_plot_encoding_space
    TODO
    :param event:
    :param encodings:
    :param images:
    :param subjects:
    :param figures:
    :param axes:
    :param collection:
    :return:
    """
    for _ in range(len(collection)):
        collection[0].remove()
        collection.pop(0)
    ind = event.ind
    if len(ind) > 1:
        try:
            datax, datay = event.artist.get_data()
        except AttributeError:
            return
        datax, datay = [datax[i] for i in ind], [datay[i] for i in ind]
        msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
        dist = np.sqrt((np.array(datax)-msx)**2+(np.array(datay)-msy)**2)
        ind = [ind[np.argmin(dist)]]
    ind = ind[0]

    # Extract figures and axes
    fig1, fig2 = figures
    ax1, ax2s = axes

    # Plot indicator circle around the selected plot
    sc = event.artist.get_sizes()[0]
    x, y = encodings[ind]
    c = ax1.scatter([x], [y], s=sc*2, ec="black", color="none", lw=1)
    collection.append(c)

    # Update image(s) on ax2s
    for i, ax in enumerate(ax2s):
        ax.clear()
        image = images[ind][i] if len(ax2s) != 1 else images[ind]
        ax.imshow(image.squeeze(), cmap="gray")
        if segmentations is not None and event.mouseevent.button != 1:
            # Overlay segmentation on right-clicks
            segmentation = segmentations[ind][i]
            segmentation = np.ma.masked_where(segmentation == 0, segmentation)
            ax.imshow(segmentation, alpha=0.30, cmap="Set1")

    # Set title if subjects were passed
    if subjects is not None:
        fig2.suptitle(subjects[ind], fontsize=18)

    # Update canvases
    disable_axes(ax1, *ax2s)
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.9)
    fig1.canvas.draw()
    fig2.canvas.draw()


def disable_axes(*axes):
    """ Calls ax.axis("off") on all passed axis objects """
    for ax in axes:
        ax.axis("off")


def interactive_plot_encoding_space(encodings,
                                    images,
                                    segmentations=None,
                                    subjects=None,
                                    color_by=None,
                                    marker_by=None,
                                    title=None,
                                    **tsne_kwargs):
    """
    Calls plot_encoding_space and opens the plot in interactive mode.
    The user may click on points and corresponding (by index on axis 0)
    images from passed array 'images' will be shown in a second figure.

    TODO

    :param encodings:
    :param images:
    :param subjects:
    :param color_by:
    :param marker_by:
    :param title:
    :param tsne_kwargs:
    :return:
    """
    # Set backend
    matplotlib.use("WebAgg")

    # Init encoding figure
    fig, ax, encodings = plot_encoding_space(encodings=encodings,
                                             color_by=color_by,
                                             marker_by=marker_by,
                                             title=title,
                                             **tsne_kwargs)

    # Init figure for showing images
    n_images = len(images[0]) if images[0].ndim == 3 else 1
    figure_2, ax2s = plt.subplots(num=fig.number+1, figsize=fig.get_size_inches(), ncols=1, nrows=n_images)
    if not isinstance(ax2s, (list, np.ndarray)):
        ax2s = [ax2s]

    # Set parameters for event handler
    params = {"encodings": encodings,
              "images": images,
              "segmentations": segmentations,
              "figures": (fig, figure_2),
              "axes": (ax, ax2s),
              "subjects": subjects,
              "collection": []}
    fig.canvas.mpl_connect('pick_event', lambda e: onpick(e, **params))

    # Start interactive mode
    disable_axes(ax, *ax2s)
    plt.show()

    # Revert to default backend
    matplotlib.use("agg")


# if __name__ == "__main__":
#     encodings = np.random.randn(100, 50)
#     color_by = np.random.randint(0, 1, 100)
#     images = np.random.randn(100, 1, 280, 480)
#     subjects = ["subject_1", "subject_2"] * 50
#     interactive_plot_encoding_space(
#         encodings=encodings,
#         color_by=color_by,
#         images=images,
#         subjects=subjects
#     )
