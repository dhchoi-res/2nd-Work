import os
from collections import namedtuple
import random
import shutil

from osgeo import gdal
import numpy as np
import geopandas as gpd
from shapely import geometry
from tqdm.notebook import tqdm
from tqdm.contrib import tzip
import folium
from branca.element import Template, MacroElement

from SIALibs.dataset.util import coordinate_ops

SceneInfo = namedtuple('SceneInfo', 'sat_name absname name ext is_anno geometry')


class SceneMgr:
    SATELLITE_NAMES = ['WV3', 'WV2', 'WV1', 'K3', 'K3A', 'SKYSAT', 'PLANETSCOPE']
    SATELLITE_COLORS = {'WV3': '#F423E8', 'WV2': '#66669C', 'WV1': '#64be99',
                        'K3': '#FAAA1E', 'K3A': '#DCDC00',
                        'SKYSAT': '#6B8E23', 'PLANETSCOPE': '#3C14DC'}

    def __init__(self, dirs=None, df=None, name_suffixes=None, img_ext=('jp2', 'tif')):
        """
        :param dirs: string(dir) or list(dirs)
        """

        assert dirs is not None or df is not None

        if dirs:
            scene_infos = list()

            for d, suffix in tzip(dirs, name_suffixes):
                paths = [os.path.join(root, name) for root, dirs, files in os.walk(d, followlinks=True)
                         for name in files if name.lower().endswith(img_ext)]
                if suffix:
                    paths = [path for path in paths if os.path.splitext(path)[0].lower().endswith(suffix)]

                for path in tqdm(paths):
                    sat_name = self._get_satellite_name(path)
                    is_anno = os.path.exists(os.path.splitext(path)[0] + '.label')

                    raster = gdal.Open(path)
                    if raster == None:
                        print(f'Can not open {path}')
                        continue

                    width, height = raster.RasterXSize, raster.RasterYSize
                    raster_list = [[0, 0], [width, 0], [width, height], [0, height]]
                    coord_transform, geo_transform = coordinate_ops.get_transform(raster)
                    geo_pts = coordinate_ops.cvt_ras_to_geo(raster_list, geo_transform=geo_transform,
                                                            coord_transform=coord_transform)
                    geo_pts = list(map(lambda x: (x[1], x[0]), geo_pts))  # lat, long -> long, lat for Folium.Polygon
                    poly = geometry.Polygon(geo_pts)

                    name, ext = os.path.splitext(os.path.basename(path))
                    scene_info = SceneInfo(sat_name=sat_name, absname=path, name=name, ext=ext, is_anno=is_anno,
                                           geometry=poly)
                    scene_infos.append(scene_info)

            self.gdf = gpd.GeoDataFrame(scene_infos)

        else:
            self.gdf = df.copy()

        # Set styles of pandas.DataFrame
        self._styles = [dict(selector="th", props=[('font-size', '12pt')]),
                        dict(selector="td", props=[('font-size', '11pt')])]

    def __len__(self):
        return self.gdf.shape[0]

    def duplicated(self):
        return self.gdf[self.gdf.name.duplicated()]

    def move_duplicated(self, dst_dir):
        df = self.gdf[self.gdf.name.duplicated()]
        duplicated_paths = df.absname.values.tolist()

        for path in tqdm(duplicated_paths):
            path = os.path.dirname(path)
            shutil.move(path, dst_dir)

    def exclude_duplicated(self, scene_mgr=None):
        if scene_mgr:
            self.gdf = self.gdf[~self.gdf.name.isin(scene_mgr.gdf.name)]
        else:
            self.gdf = self.gdf[~self.gdf.name.duplicated()]

    def not_annotated(self):
        return self.gdf[self.gdf.is_anno == False]

    def counts_by_regions(self, is_only_anno=False, is_plot=False):
        df = self.gdf
        if is_only_anno:
            df = self.gdf[self.gdf.is_anno == True]

        df = df.groupby('region').size().reset_index(name='counts')

        if is_plot:
            return df.plot(x='region', y='counts', kind='bar')
        else:
            return df.style.set_table_styles(self._styles)

    def counts_by_sat_names(self, is_plot=False):
        df = self.gdf.groupby('sat_name').size().reset_index(name='counts')

        if is_plot:
            return df.plot(x='sat_name', y='counts', kind='bar')
        else:
            return df.style.set_table_styles(self._styles)

    def counts_by_annotated(self):
        return self.gdf[self.gdf.is_anno == True].shape[0]

    def counts_by_inferenced(self):
        return self.gdf[self.gdf.is_infer == True].shape[0]

    def get_not_inferenced(self):
        return SceneMgr(df=self.gdf[self.gdf.is_infer == False])

    def get_absname(self, name):
        return self.gdf[self.gdf.name == name].absname.values[0]

    def split_train_and_test(self, test_file_ratio=0.2):
        train_filenames, test_filenames = list(), list()

        for region in self.gdf.region.unique():
            region_df = self.gdf[self.gdf.region == region]
            for sat_name in region_df.sat_name.unique():
                filenames = region_df[region_df.sat_name == sat_name].name.values.tolist()
                random.seed(42)
                random.shuffle(filenames)
                n_test_file = round(len(filenames) * test_file_ratio)
                test_filenames += filenames[:n_test_file]
                train_filenames += filenames[n_test_file:]

        return train_filenames, test_filenames

    def split_n(self, n, dst_paths, is_move=False):
        if not isinstance(dst_paths, list):
            dst_paths = [os.path.join(dst_paths, str(idx)) for idx in range(n)]
            for dst_path in dst_paths:
                os.makedirs(dst_path, exist_ok=True)

        for region in tqdm(self.gdf.region.unique()):
            region_df = self.gdf[self.gdf.region == region]
            for sat_name in tqdm(region_df.sat_name.unique()):
                filenames = region_df[region_df.sat_name == sat_name].absname.values.tolist()
                random.seed(42)
                random.shuffle(filenames)

                splited_filenames_list = np.array_split(filenames, n)
                for idx, (splited_filenames, dst_path) in enumerate(tzip(splited_filenames_list, dst_paths)):
                    for filename in tqdm(splited_filenames):
                        src_dir = os.path.dirname(filename)
                        dst_dir = os.path.join(dst_path, src_dir[src_dir.rfind('scenes'):])
                        if not os.path.exists(dst_dir):
                            if is_move:
                                shutil.move(src_dir, os.path.join(dst_path, dst_dir))
                            else:
                                shutil.copytree(src_dir, os.path.join(dst_path, dst_dir))

    def sample(self, sample_ratio):
        sampled_df = gpd.GeoDataFrame()
        for region in self.gdf.region.unique():
            region_df = self.gdf[self.gdf.region == region]
            for sat_name in region_df.sat_name.unique():
                sat_name_df = region_df[region_df.sat_name == sat_name]
                sampled_df = sampled_df.append(sat_name_df.sample(frac=sample_ratio))

        return sampled_df

    def save_csv(self, filepath):
        self.gdf.to_csv(filepath, mode='w')

    def copy_scenes(self, dst_dir):
        paths = self.gdf.absname.unique()

        for path in tqdm(paths):
            src_path = os.path.dirname(path)
            dst_path = os.path.join(dst_dir, src_path[src_path.rfind('scenes'):])
            shutil.copytree(src_path, dst_path)

    def copy_labels(self, dst_dir):
        paths = self.gdf.absname.values.tolist()
        for path in tqdm(paths):
            path = os.path.splitext(path)[0] + '.label'
            if os.path.exists(path):
                shutil.copy(path, dst_dir)

    def _get_satellite_name(self, path):
        path = path.upper()

        sat_match_indices = [path.rfind(sat_name) for sat_name in self.SATELLITE_NAMES]
        sat_idx = int(np.argmax(sat_match_indices))
        if sat_match_indices[sat_idx] != -1:
            return self.SATELLITE_NAMES[sat_idx]
        else:
            return 'UNKNOWN'

    def vis_on_map(self, out_path=''):
        m = folium.Map()

        sat_names = self.gdf.sat_name.unique()
        for sat_name in sat_names:
            sat_gdf = self.gdf[self.gdf.sat_name == sat_name]
            polys = sat_gdf.geometry.values
            names = sat_gdf.name.values.tolist()
            for poly, name in zip(polys, names):
                folium.Polygon(
                    locations=list(poly.exterior.coords),
                    color=self.SATELLITE_COLORS[sat_name],
                    weight=2,
                    fill=True,
                    fill_opacity=0.1,
                    fill_color=self.SATELLITE_COLORS[sat_name],
                    tooltip=name
                ).add_to(m)

        legend = self.get_legend()
        m.get_root().add_child(legend)

        if out_path:
            m.save(out_path)

        return m

    def get_legend(self):
        template_frond = """
        {% macro html(this, kwargs) %}

        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>jQuery UI Draggable - Default functionality</title>
          <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

          <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
          <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

          <script>
          $( function() {
            $( "#maplegend" ).draggable({
                            start: function (event, ui) {
                                $(this).css({
                                    right: "auto",
                                    top: "auto",
                                    bottom: "auto"
                                });
                            }
                        });
        });

          </script>
        </head>
        <body>


        <div id='maplegend' class='maplegend' 
            style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
             border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

        <div class='legend-title'> Sensor </div>
        <div class='legend-scale'>
          <ul class='legend-labels'>
        """

        template_end = """
          </ul>
        </div>
        </div>

        </body>
        </html>

        <style type='text/css'>
          .maplegend .legend-title {
            text-align: left;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 90%;
            }
          .maplegend .legend-scale ul {
            margin: 0;
            margin-bottom: 5px;
            padding: 0;
            float: left;
            list-style: none;
            }
          .maplegend .legend-scale ul li {
            font-size: 80%;
            list-style: none;
            margin-left: 0;
            line-height: 18px;
            margin-bottom: 2px;
            }
          .maplegend ul.legend-labels li span {
            display: block;
            float: left;
            height: 16px;
            width: 30px;
            margin-right: 5px;
            margin-left: 0;
            border: 1px solid #999;
            }
          .maplegend .legend-source {
            font-size: 80%;
            color: #777;
            clear: both;
            }
          .maplegend a {
            color: #777;
            }
        </style>
        {% endmacro %}"""

        legend = []
        sat_names = self.gdf.sat_name.unique()
        for sat_name in sat_names:
            legend.append("\t\t<li><span style='background:{};opacity:0.7;'></span>{}</li>".format(
                self.SATELLITE_COLORS[sat_name],
                sat_name
            ))

        template = template_frond + ''.join(legend) + template_end
        macro = MacroElement()
        macro._template = Template(template)

        return macro
