import napari
from magicgui import magicgui
from napari.layers import Image, Layer
from napari.types import LayerDataTuple
import numpy as np
from typing import List

# Custom napari function that splits channels
@magicgui(call_button="Split channels")
def split(viewer: napari.Viewer, layer: Layer) -> None:
    image = layer.data
    name = layer.name
    del viewer.layers[name]  # Delete the old layer

    # Determine the channel dimension (assume smallest shape for channels)
    channel_dim = np.argmin(image.shape)
    image = np.moveaxis(image, channel_dim, 1)
    n_channels = image.shape[1]

    # First channel is always raw
    raw = image[:, 0]

    # Add the "raw" layer
    viewer.add_image(raw, name='raw', opacity=0.5, scale=[3.2, 1, 1])

    # Add segmentation channels with generic names
    for i in range(1, n_channels):
        seg = image[:, i]
        viewer.add_labels(seg.astype('uint16'), name=f'seg{i}', opacity=0.5, scale=[3.2, 1, 1])

# Add widget and fire up napari
viewer = napari.Viewer()
viewer.window.add_dock_widget(split)
napari.run()