import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
import dota_utils as util
from collections import defaultdict
import cv2
from typing import override
from DOTA import DOTA


# copied from DOTA
def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def _filter_by_extensions(items: list[str], exts: list[str]) -> list[str]:
    def ends_with_any_extension(item: str) -> bool:
        return any(item.endswith(ext) for ext in exts)
    return [l for l in items if ends_with_any_extension(l)]

class DOTAUltralytics(DOTA):
    """
    Modified version of DOTA class to fit with ultralytics' folder structure
    """
    def __init__(self, basepath, labels_basepath='labelTxt', images_base_path='images', useoriginal=False):
        # DOTA.__init__(self, basepath)
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, labels_basepath)
        self.imagepath = os.path.join(basepath, images_base_path)

        self.imgpaths = util.GetFileFromThisRootDir(self.imagepath)
        self.imgpaths = _filter_by_extensions(self.imgpaths, ['.jpg', '.png', '.jpeg'])
        self.imglist = [util.custombasename(x) for x in self.imgpaths]

        self.annspaths: list[str] = util.GetFileFromThisRootDir(self.labelpath)
        if useoriginal:
            self.annspaths = [x for x in self.annspaths if '_original' in x]
        else:
            self.annspaths = [x for x in self.annspaths if '_original' not in x]
        self.annspaths = _filter_by_extensions(self.annspaths, ['.txt'])
        self.annslist = [util.custombasename(x) for x in self.annspaths]

        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)

        self.useoriginal = useoriginal

        assert not any(x.endswith('.cache') for x in self.imgpaths + self.annspaths)

        self.createIndex()

    @override
    def createIndex(self):
        assert len(self.imglist) == len(set(self.imgpaths)), "Expected amount of ids to match files. Naming collisions?"
        assert len(self.annspaths) == len(set(self.annslist)), "Expected amount of ids to match files. Naming collisions?"

        assert set(self.annslist).issubset(set(self.imglist))
        assert len(self.imglist) > len(self.annslist)

        for filename in self.imgpaths:
            imgid = util.custombasename(filename)
            if imgid not in self.annslist:
                continue
            idx = self.annslist.index(imgid)
            annsfile = self.annspaths[idx]
            objects = util.parse_dota_poly(annsfile)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)
    @override
    def loadImgs(self, imgids=[], ext='png'):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            idx = self.imglist.index(imgid)
            # filename = os.path.join(self.imagepath, imgid + '.' + ext)
            filename = self.imgpaths[idx]
            print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs

if __name__ == "__main__":
    d = DOTAUltralytics('DOTAv1', labels_basepath='labels', useoriginal=True)
    print(len(d.imgpaths))
    print(d.ImgToAnns['P0000'])
    img = d.loadImgs(['P0000'])[0]
    cv2.imshow('img', img)
    cv2.waitKey(0)