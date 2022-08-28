## Cloud Native Data in Geosciences.

There is a shift is happening in the way we use Earth Science data to do new research. Cloud data storage technologies have advanced at a fast pace that we now can find and explore massive amounts of data online via the web browser. At the same time we can find online platforms with specialized software and hardware that offer general data science and machine learning tools to explore these online datasets.

With these advances it is easier to foster collaborations, promote data-driven discovery, scientific innovation, transparency and reproducibility.

***

## Cloud Data Types

We will be reviewing the next set of data types
* [GeoJSON](https://geojson.org/)
* [Cloud Optimized GeoTiff (COG)](https://www.cogeo.org/)
* [Cloud Optimized Point Clouds (COPC)](https://copc.io/)

***

### GeoJSON

[GeoJSON](https://en.wikipedia.org/wiki/GeoJSON) is a format that supports encoding a wide variety of gegraphic data structures using JavaScript Object Notation (JSON) [[RFC7159](https://datatracker.ietf.org/doc/html/rfc7159)].  A
GeoJSON object may represent a region of space (a Geometry), a
spatially bounded entity (a Feature), or a list of Features (a
FeatureCollection).

It has the following format:
```
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [125.6, 10.1]
  },
  "properties": {
    "name": "Dinagat Islands"
  }
}

```

Where the "type" can be any of the following:

| Type | Geometry |
| :--: | :--:     |
|Point   |  ![Point](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/SFA_Point.svg/51px-SFA_Point.svg.png)|
| LineString   |  ![Linestring](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/SFA_LineString.svg/51px-SFA_LineString.svg.png)|
|Polygon   |   ![Polygon](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/SFA_Polygon.svg/51px-SFA_Polygon.svg.png) |
|MultiPoint|  ![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/SFA_MultiPoint.svg/51px-SFA_MultiPoint.svg.png)  |
| MultiLineString  |    ![MultiLineString](https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/SFA_MultiLineString.svg/51px-SFA_MultiLineString.svg.png)|
|MultiPolygon   |  ![MultiPolygon](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/SFA_MultiPolygon_with_hole.svg/51px-SFA_MultiPolygon_with_hole.svg.png)
| GeometryCollection | ![GeometryCollection](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/SFA_GeometryCollection.svg/51px-SFA_GeometryCollection.svg.png) |

Read more about the [GeoJSON format](https://datatracker.ietf.org/doc/html/rfc7946).

### COG


The TIFF file format (Tagged Image File Format) is a very old format, dating back to 1992, which is great for high-resolution, verbatim raster images. It’s still used a bit in high-end photography, but has really grown a second life in cartography: a variation called GeoTIFF is used to share satellite images and other satellite data.

The GeoTIFF file format has long been thought of as only suitable for raw data: if you wanted to display it on a map, you’d convert it into tiles. If you wanted a static image, you’d render it into a PNG or JPEG. But Cloud-Optimized GeoTIFF means that GeoTIFFs can be a bit more accessible than they used to be.

A [Cloud Optimized GeoTIFF (COG)](https://www.cogeo.org/) is a regular [GeoTIFF file](https://en.wikipedia.org/wiki/GeoTIFF), aimed at being hosted on a HTTP file server, with an internal organization that enables more efficient workflows on the cloud. It does this by leveraging the ability of clients issuing ​HTTP GET range requests to ask for just the parts of a file they need.


### COPC






### Spatio Temporal Asset Catalogs (STAC)









## References
* [Gentemann, C. L., Holdgraf, C., Abernathey, R., Crichton, D., Colliander, J., Kearns, E. J., et al. (2021). Science storms the cloud. AGU Advances, 2, e2020AV000354](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020AV000354)
* [Ryan P. Abernathey _et al._ (2021) Cloud-Native Repositories for Big Scientific Data. Computing in Science and Engineering. IEEE Computer Society](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9354557)

***
Created: 08/18/2022;
Updated: 08/22/2022

Carlos Lizárraga.
UA Data Science Institute.
