import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def _quiet(msg):
    pass

_info = _quiet
_warn = _quiet

def set_renderer_logger(info, warn):
    global _info, _warn
    if info is not None:
        _info = info
    if warn is not None:
        _warn = warn

def load_obj(obj_path):
    if not os.path.isfile(obj_path):
        raise IOError("Cannot open %s" % obj_path)

    obj_base_path = os.path.dirname(os.path.abspath(obj_path))
    obj = {
        'materials': {},
    }
    vertices = []
    uvs = []

    faces = []

    with open(obj_path) as f:
        _info("Loading %s" % obj_path)

        for line in f:
            if line.startswith("mtllib "):
                # Materials
                mtl_file = "".join(line.split()[1:]).strip()
                obj['materials'].update(load_mtl(mtl_file, obj_base_path))
            elif line.startswith("v "):
                # Vertices
                vertices.append([list(map(float, line.split()[1:4]))])
            elif line.startswith("vt "):
                # UVs
                uvs.append([list(map(float, line.split()[1:3]))])
            elif line.startswith("usemtl "):
                if 'materials_idx' not in obj:
                    obj['materials_idx'] = list(obj['materials'].keys())

                mtl_name = "".join(line.split()[1:]).strip()
                if not mtl_name in obj['materials']:
                    raise Exception("%s material is missing" % mtl_name)

                current_material = obj['materials_idx'].index(mtl_name)
            
            elif line.startswith("f "):
                a,b,c = line.split()[1:]
                av, at = map(int, a.split("/")[0:2])
                bv, bt = map(int, b.split("/")[0:2])
                cv, ct = map(int, c.split("/")[0:2])
                
                faces.append(((av, bv, cv), (at, bt, ct), current_material)) 


    obj['vertices'] = np.array(vertices)
    obj['uvs'] = np.array(uvs)
    obj['faces'] = faces

    return obj

def load_mtl(mtl_file, obj_base_path):
    mtl_file = os.path.join(obj_base_path, mtl_file)

    if not os.path.isfile(mtl_file):
        raise IOError("Cannot open %s" % mtl_file)
    
    mats = {}
    current_mtl = ""

    with open(mtl_file) as f:
        for line in f:
            if line.startswith("newmtl "):
                current_mtl = "".join(line.split()[1:]).strip()
            elif line.startswith("map_Kd ") and current_mtl:
                map_kd_filename = "".join(line.split()[1:]).strip()
                map_kd = os.path.join(obj_base_path, map_kd_filename)
                if not os.path.isfile(map_kd):
                    raise IOError("Cannot open %s" % map_kd)
                
                _info("Loading %s" % map_kd_filename)
                with rasterio.open(map_kd, 'r') as r:
                    mats[current_mtl] = reshape_as_image(r.read())
    return mats

set_renderer_logger(print, print)

obj = load_obj("/datasets/brighton2/odm_texturing_25d/odm_textured_model_geo.obj")
_info("Number of faces: %s" % len(obj['faces']))