import napari
from magicgui import magicgui
from napari.layers import Image, Layer
from napari.types import LayerDataTuple
import numpy as np
from typing import List


# custom napari function that splits channels
@magicgui(call_button="Split channels")
def split(viewer: napari.Viewer, layer: Layer) -> List[LayerDataTuple]:
    image = layer.data
    name = layer.name
    del viewer.layers[name]  # delete old layer

    channel_dim = np.argmin(image.shape)
    image = np.moveaxis(image, channel_dim, 1)
    n_channels = image.shape[1]

    # first channel is always raw
    raw = image[:, 0]

    return_object = [
        (raw,
         dict(name='raw', opacity=0.5, scale=[3.2, 1, 1]),
         'image'),
    ]

    # subsequent channels are all segmentation channels, give them generic names
    for i in range(n_channels-1):
        seg = image[:, i+1]
        return_object.append((
            seg.astype('uint16'),
            dict(name=f'seg{i+1}',
                 opacity=0.5,
                 scale=[3.2, 1, 1],
                 ),
            'labels'),
        )
    return return_object


# add widget and fire up napari
viewer = napari.Viewer()
viewer.window.add_dock_widget(split)
napari.run()
