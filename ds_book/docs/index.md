

<!-- #region -->
# Deep Learning with TensorFlow: <br> Tutorials for modeling LULC.


[**Development Seed**](https://developmentseed.org/)

<!-- #endregion -->

These materials are designed to provide TensorFlow expertise via tutorials and science support, vis a vis suggestions for acquisition and processing of data inputs, training, testing, and evaluation of TensorFlow models as well as different TensorFlow / deep learning techniques demoed in Colab notebooks using real data. 

The content of this workshop assumes general familiarity with geospatial data such as satellite imagery, raster and vector formats, file formats such as [GeoTIFF](https://earthdata.nasa.gov/esdis/eso/standards-and-references/geotiff) and [GeoJSON](https://geojson.org/), the [python](https://www.python.org/) programming language and [Google Colab](https://research.google.com/colaboratory/). Having knowledge of [numpy](https://numpy.org/), [rasterio](https://rasterio.readthedocs.io/en/latest/), [geopandas](https://geopandas.org/en/stable/) and [sci-kit learn](https://scikit-learn.org/stable/) is a plus. 


```{admonition} Links
:class: tip

- Jupyter Book: 
[https://developmentseed.github.io/tensorflow-eo-training/docs/index.html)

```


```{admonition} How to run the notebook code
:class: important

A major advantage of executable books is that the reader may enjoy running the source code, modifying them and playing around. No downloading, installation or configuration are required. Simply go to 

[https://developmentseed.github.io/tensorflow-eo-training/docs/index.html](https://developmentseed.github.io/tensorflow-eo-training/docs/index.html),

and in the left menu select any topic, click the "rocket" icon at the top right of the screen, and choose “Colab." This will launch the page in a virtual runtime environment hosted by Google. From there, the code can be run using a free GPU.

For local running, the code for each topic in the form of 
[Jupyter](https://jupyter.org) notebooks can be downloaded by clicking the "arrow-down" icon at the top right of the screen. 

```

```{admonition} How to access the data
:class: important

Ideally, all data would be ingested from a storage bucket on Google Cloud Platform, however, input/output to Google Colab from cloud storage is prohibitively slow and fragile. For the purpose of this workshop, we have shared a Google Drive folder with participants. To use its contents during the workshop, please right click on the folder from Google Drive and select `Add shortcut to Drive` and select your `My Drive` folder as the destination. This will allow you to programmatically read the data via a path that mimics your personal drive rather than your shared drive. 

```

```{admonition} $~$
Built with [Jupyter Book
2.0](https://beta.jupyterbook.org/intro.html) tool set, as part of the
[ExecutableBookProject](https://ebp.jupyterbook.org/en/latest/).  
```



ISBN: *(tbd)


