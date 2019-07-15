import rasterio
from matplotlib import pyplot
from rasterio.plot import show
import os
import geopandas


fig, ax = pyplot.subplots(figsize=(20, 10))


path = geopandas.datasets.get_path('naturalearth_lowres')
earth_info = geopandas.read_file(path)

forest_data_path = "/Users/anmartin/Projects/summer_project/gym-orekit/forest_data.tiff"
src = rasterio.open(forest_data_path)
print(src.count, src.bounds, src.crs)
show(src, cmap="Greens", ax=ax)

earth_info.plot(ax=ax, facecolor='none', edgecolor='red')
pyplot.show()