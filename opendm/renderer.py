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

class Bbox:
    def __init__(self):
        self.xmin = np.inf
        self.xmax = -np.inf
        self.ymin = np.inf
        self.ymax = -np.inf
    
    def __str__(self):
        return "[(%s, %s), (%s, %s)]" % (self.xmin, self.ymin, self.xmax, self.ymax)
    
    def __eq__(self, other):
        if isinstance(other, Bbox):
            return  self.xmin == other.xmin and \
                    self.ymin == other.ymin and \
                    self.xmax == other.xmax and \
                    self.ymax == other.ymax 
        return False
    
    def area(self):
        return self.width() * self.height()

    def width(self):
        return self.xmax - self.xmin
    
    def height(self):
        return self.ymax - self.ymin
    
def set_renderer_logger(info, warn):
    global _info, _warn
    if info is not None:
        _info = info
    if warn is not None:
        _warn = warn

def roi_transform(bbox, res_px):
    return np.array([[res_px, 0.0, 0.0, -bbox.xmin * res_px],
                    [0.0, -res_px, 0.0, bbox.ymax * res_px],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

def load_obj(obj_path):
    if not os.path.isfile(obj_path):
        raise IOError("Cannot open %s" % obj_path)

    obj_base_path = os.path.dirname(os.path.abspath(obj_path))
    obj = {
        'materials': {},
    }
    vertices = []
    uvs = []

    faces = {}
    current_material = "_"

    with open(obj_path) as f:
        _info("Loading %s" % obj_path)

        for line in f:
            if line.startswith("mtllib "):
                # Materials
                mtl_file = "".join(line.split()[1:]).strip()
                obj['materials'].update(load_mtl(mtl_file, obj_base_path))
            elif line.startswith("v "):
                # Vertices
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith("vt "):
                # UVs
                uvs.append([list(map(float, line.split()[1:3]))])
            elif line.startswith("usemtl "):
                mtl_name = "".join(line.split()[1:]).strip()
                if not mtl_name in obj['materials']:
                    raise Exception("%s material is missing" % mtl_name)

                current_material = mtl_name
            elif line.startswith("f "):
                a,b,c = line.split()[1:]
                av, at = map(int, a.split("/")[0:2])
                bv, bt = map(int, b.split("/")[0:2])
                cv, ct = map(int, c.split("/")[0:2])
                
                if current_material not in faces:
                    faces[current_material] = []

                faces[current_material].append(((av, bv, cv), (at, bt, ct))) 

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

                # mats[current_mtl] = True
                with rasterio.open(map_kd, 'r') as r:
                    mats[current_mtl] = reshape_as_image(r.read())
    return mats

def compute_bounds(obj):
    bbox = Bbox()
    for v in obj['vertices']:
        bbox.xmin = min(bbox.xmin, v[0])
        bbox.xmax = max(bbox.xmax, v[0])
        bbox.ymin = min(bbox.ymin, v[1])
        bbox.ymax = max(bbox.ymax, v[1])
    return bbox

def max_range(dtype):
    try:
        return np.iinfo(dtype).max
    except ValueError:
        return np.finfo(dtype).max


def render_orthophoto(input_objs, resolution=5.0):
    """Render OBJ files
    :param input_objs (str[]) array of paths to .OBJ models
    :param resolution (float) resolution of resulting raster in cm/px
    """
    if isinstance(input_objs, str):
        input_objs = [input_objs]
    
    primary = True
    bounds = None

    res_px = 100.0 / resolution 

    for in_obj in input_objs:
        obj = load_obj(in_obj)
        _info("Number of faces: %s" % len(obj['faces']))

        b = compute_bounds(obj)
        if primary:
            bounds = b
        else:
            # Quick check
            if bounds != b:
                raise Exception("Bounds between models must all match, but they don't.")

        _info("Model bounds: %s" % str(b))
        _info("Model area: %s m2" % round(b.area(), 2))

        height = int(np.ceil(res_px * b.height()))
        width = int(np.ceil(res_px * b.width()))

        _info("Model resolution: %sx%s" % (width, height))

        if height <= 0:
            _warn("Orthophoto has negative height, forcing height = 1")
            height = 1.0
        
        if width <= 0:
            _warn("Orthophoto has negative width, forcing height = 1")
            width = 1.0
        
        transform = roi_transform(bounds, res_px)
        _info("Translating and scaling mesh...")
        h_vertices = np.hstack((obj['vertices'], np.ones((len(obj['vertices']), 1))))
        obj['vertices'] = h_vertices.dot(transform.T)[:,:-1]

        _info("Rendering the orthophoto...")

        # Used to keep track of the global face index
        faceOff = 0 # TODO: necessary?

        # Iterate over each part of the mesh (one per material)
        data_type = None
        current_band_index = 0 # TODO: necessary?
        texture = None

        bands = []
        alpha_band = np.zeros((width, height))
        materials_idx = list(obj['materials'].keys())

        for mat in materials_idx:
            texture = obj['materials'][mat]

            # First material determines the data type
            if mat == materials_idx[0]:
                # Init orthophoto

                if primary:
                    data_type = texture.dtype
                elif data_type != texture.dtype:
                    # Try to convert
                    # TODO!
                    pass
                
                _info("Texture bands: %s" % texture.shape[2])
                _info("Texture type: %s" % texture.dtype)

                fill_value = max_range(data_type)
                for n in range(texture.shape[2]):
                    band = np.full((width, height), fill_value)
                    bands.append(band)

            # Render

            # For each face in the current material
            for f in obj['faces'][mat]:
                print(f)
        
        if texture is not None:
            current_band_index += texture.shape[2]
        
        primary = False


set_renderer_logger(print, print)

render_orthophoto(["/datasets/brighton2/odm_texturing_25d/odm_textured_model_geo.obj"], 5.0)


# TODO - Compute area boundaries
# 