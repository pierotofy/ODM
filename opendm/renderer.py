import rasterio
# from rasterio.plot import reshape_as_image
import numpy as np
import warnings
import os

# Originally ported from https://github.com/OpenDroneMap/odm_orthophoto

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def _quiet(msg):
    pass

_info = _quiet
_warn = _quiet

eps = np.finfo(float).eps
eps2 = eps * eps

class Bbox:
    def __init__(self):
        self.xmin = np.inf
        self.xmax = -np.inf
        self.ymin = np.inf
        self.ymax = -np.inf
    
    def __str__(self):
        return "%s" % (self.as_array())
    
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

    def as_array(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])
    
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
        # 'materials_idx': {},
        'faces': []
    }
    vertices = []
    uvs = []

    faces = []
    current_material = None

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
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith("usemtl "):
                mtl_name = "".join(line.split()[1:]).strip()
                if not mtl_name in obj['materials']:
                    raise Exception("%s material is missing" % mtl_name)
                if not 'materials_idx' in obj:
                    obj['materials_idx'] = list(obj['materials'].keys())

                current_material = obj['materials_idx'].index(mtl_name)
            elif line.startswith("f "):
                a,b,c = line.split()[1:]
                av, at = map(int, a.split("/")[0:2])
                bv, bt = map(int, b.split("/")[0:2])
                cv, ct = map(int, c.split("/")[0:2])
                
                # if current_material not in faces:
                #     faces[current_material] = []

                #faces[current_material].append(((av - 1, bv - 1, cv - 1), (at - 1, bt - 1, ct - 1))) 
                #faces.append((current_material, *vertices[av -1], *vertices[bv - 1], *vertices[cv - 1], *uvs[at - 1], *uvs[bt - 1], *uvs[ct - 1]))
                obj['faces'].append((current_material, av - 1, bv - 1, cv - 1, at - 1, bt - 1, ct - 1))

    obj['vertices'] = np.array(vertices)
    obj['uvs'] = np.array(uvs)

    # for f in faces:
    #     obj['faces'][f] = np.array(faces[f])

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
                    mats[current_mtl] = r.read() #reshape_as_image(r.read())
    return mats

def compute_bounds(obj):
    bbox = Bbox()
    bbox.xmin = np.min(obj['vertices'][:,0])
    bbox.xmax = np.max(obj['vertices'][:,0])
    bbox.ymin = np.min(obj['vertices'][:,1])
    bbox.ymax = np.max(obj['vertices'][:,1])
    return bbox

def max_range(dtype):
    try:
        return np.iinfo(dtype).max
    except ValueError:
        return np.finfo(dtype).max

# def is_sliver_polygon(v1, v2, v3):
#     dummy_vec = np.cross(v1 - v2, v3 - v2)
#     return eps2 >= dummy_vec.dot(dummy_vec) / 2.0

def filter_sliver_polygons(faces):
    v1 = faces[:,1:4]
    v2 = faces[:,4:7]
    v3 = faces[:,7:10]
    dummy_vec = np.cross(v1 - v2, v3 - v2)
    return faces[np.sum(dummy_vec*dummy_vec, axis=1) / 2.0 > eps2]

# TODO: SLOW
def get_barycentric_coordinates(v1, v2, v3, x, y):
    # Diff along y
    y2y3 = v2[1] - v3[1]
    y1y3 = v1[1] - v3[1]
    y3y1 = v3[1] - v1[1]
    yy3 = y - v3[1]

    # Diff along x
    x3x2 = v3[0] - v2[0]
    x1x3 = v1[0] - v3[0]
    xx3 = x - v3[0]

    norm = y2y3 * x1x3 + x3x2 * y1y3
    l1 = (y2y3 * xx3 + x3x2 * yy3) / norm
    l2 = (y3y1 * xx3 + x1x3 * yy3) / norm
    l3 = 1.0 - l1 - l2
    return (l1, l2, l3)

def render_orthophoto(input_objs, resolution, output):
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

        # Iterate over each part of the mesh (one per material)
        data_type = None
        current_band_index = 0
        texture = None

        all_bands = []
        alpha_band = None
        depth = np.full((width, height), -np.inf, dtype=np.float32)
        # materials_idx = list(obj['materials'].keys())

        faces = []
        vertices = obj['vertices']
        uvs = obj['uvs']

        # Create faces numpy struct
        for f in obj['faces']:
            faces.append((f[0], *vertices[f[1]],*vertices[f[2]],*vertices[f[3]],
                                *uvs[f[4]], *uvs[f[5]], *uvs[f[6]]))
        faces = np.array(faces)

        # Remove sliver polygons (if any)
        num_faces_before = len(faces)
        faces = filter_sliver_polygons(faces)
        num_faces = len(faces)

        if num_faces < num_faces_before:
            _warn("Removed %s sliver polygons" % (num_faces_before - num_faces))
        
        mat_idx = 0
        for mat in obj['materials_idx']:
            texture = obj['materials'][mat]

            # First material determines the data type
            if mat == obj['materials_idx'][0]:
                # Init orthophoto

                if primary:
                    data_type = texture.dtype
                elif data_type != texture.dtype:
                    # Try to convert
                    # TODO!
                    pass
                
                _info("Texture bands: %s" % texture.shape[0])
                _info("Texture type: %s" % texture.dtype)

                bands = np.full((texture.shape[0], width, height), max_range(data_type), dtype=data_type)
                alpha_band = np.zeros((width, height), dtype=data_type)
                all_bands.append(bands)

            # Render

            rows, cols = map(float, texture.shape[1:])
            num_channels = texture.shape[0]

            # For each face in the current material
            mat_faces = faces[faces[:,0] == mat_idx]

            for f in mat_faces:
                # Draw textured triangle

                # v1i, v2i, v3i = f[0]
                # v1 = obj['vertices'][v1i]
                # v2 = obj['vertices'][v2i]
                # v3 = obj['vertices'][v3i]

                v = f[1:10]
                t = f[10:]

                v1 = v[0:3]
                v2 = v[3:6]
                v3 = v[6:9]

                v1t = t[0:2]
                v2t = t[2:4]
                v3t = t[4:6]

                # Top point row and column positions
                top_r = top_c = None
                # Middle point row and column positions
                mid_r = mid_c = None
                # Bottom point row and column positions
                bot_r = bot_c = None

                if v1[1] < v2[1]:
                    if v1[1] < v3[1]:
                        if v2[1] < v3[1]:
                            # 1 -> 2 -> 3
                            top_r, top_c = v1[1], v1[0]
                            mid_r, mid_c = v2[1], v2[0]
                            bot_r, bot_c = v3[1], v3[0]
                        else:
                            # 1 -> 3 -> 2
                            top_r, top_c = v1[1], v1[0]
                            mid_r, mid_c = v3[1], v3[0]
                            bot_r, bot_c = v2[1], v2[0]
                    else:
                        # 3 -> 1 -> 2
                        top_r, top_c = v3[1], v3[0]
                        mid_r, mid_c = v1[1], v1[0]
                        bot_r, bot_c = v2[1], v2[0]
                else: # v2[1] <= v1[1]
                    if v2[1] < v3[1]:
                        if v1[1] < v3[1]:
                            # 2 -> 1 -> 3
                            top_r, top_c = v2[1], v2[0]
                            mid_r, mid_c = v1[1], v1[0]
                            bot_r, bot_c = v3[1], v3[0]
                        else:
                            # 2 -> 3 -> 1
                            top_r, top_c = v2[1], v2[0]
                            mid_r, mid_c = v3[1], v3[0]
                            bot_r, bot_c = v1[1], v1[0]
                    else:
                        # 3 -> 2 -> 1
                        top_r, top_c = v3[1], v3[0]
                        mid_r, mid_c = v2[1], v2[0]
                        bot_r, bot_c = v1[1], v1[0]
                # rows = float(height)
                # cols = float(width)

                # General appreviations:
                # ---------------------
                # tm : Top(to)Middle.
                # mb : Middle(to)Bottom.
                # tb : Top(to)Bottom.
                # c  : column.
                # r  : row.
                # dr : DeltaRow, step value per row.

                # The step along column for every step along r. Top to middle.
                ctmdr = None
                # The step along column for every step along r. Top to bottom.
                ctbdr = None
                # The step along column for every step along r. Middle to bottom.
                cmbdr = None

                ctbdr = (bot_c - top_c) / (bot_r - top_r)

                # The current column position, from top to middle.
                ctm = top_c
                # The current column position, from top to bottom.
                ctb = top_c

                # Check for vertical line between middle and top.
                if eps < mid_r - top_r:
                    ctmdr = (mid_c - top_c) / (mid_r - top_r)

                    # The first pixel row for the bottom part of the triangle.
                    rq_start = max(int(np.floor(top_r + 0.5)), 0)
                    
                    # The last pixel row for the top part of the triangle.
                    rq_end = min(int(np.floor(mid_r + 0.5)), height)

                    for rq in range(rq_start, rq_end):
                        # Set the current column positions.
                        ctm = top_c + ctmdr * (float(rq)+0.5-top_r)
                        ctb = top_c + ctbdr * (float(rq)+0.5-top_r)

                        # The first pixel column for the current row
                        cq_start = max(int(np.floor(0.5 + min(ctm, ctb))), 0)
                        
                        # The last pixel column for the current row.
                        cq_end = min(int(np.floor(0.5 + max(ctm, ctb))), width)

                        for cq in range(cq_start, cq_end):
                            l1, l2, l3 = get_barycentric_coordinates(v1, v2, v3, float(cq) + 0.5, float(rq) + 0.5)
                            if l1 < 0 or l2 < 0 or l3 < 0:
                                continue

                            z = v1[2] * l1 + v2[2] * l2 + v3[2] * l3

                            depth_value = depth[cq,rq]
                            if z < depth_value:
                                # Current is behind another, don't draw
                                continue

                            # The uv values of the point
                            u = v1t[0] * l1 + v2t[0] * l2 + v3t[0] * l3
                            v = v1t[1] * l1 + v2t[1] * l2 + v3t[1] * l3
                            
                            # Render pixel
                            s = u * float(cols)
                            t = (1.0 - v) * float(rows)

                            dl, left_f = np.modf(s)
                            dr = 1.0 - dl
                            dt, top_f = np.modf(t)
                            db = 1.0 - dt

                            left = int(left_f)
                            top = int(top_f)

                            for i in range(0, num_channels):
                                value = 0.0

                                tl = float(texture[i,top,left])
                                tr = float(texture[i,top,left+1])
                                bl = float(texture[i,top+1,left])
                                br = float(texture[i,top+1,left+1])

                                value += tl * dr * db
                                value += tr * dl * db
                                value += bl * dr * dt
                                value += br * dl * dt

                                bands[current_band_index + i,cq,rq] = value

                            # Increment the alpha band if the pixel was visible for this band
                            # the final alpha band will be set to 255 if alpha == num bands
                            # (all bands have information at this pixel)
                            alpha_band[cq,rq] += num_channels

                            # Update depth buffer
                            depth[cq,rq] = z
                
                if eps < bot_r - mid_r:
                    cmbdr = (bot_c - mid_c) / (bot_r - mid_r)

                    # The current column position, from middle to bottom.
                    cmb = mid_c

                    # The first pixel row for the bottom part of the triangle.
                    rq_start = max(int(np.floor(mid_r + 0.5)), 0)
                    
                    # The last pixel row for the top part of the triangle.
                    rq_end = min(int(np.floor(bot_r + 0.5)), height)

                    for rq in range(rq_start, rq_end):
                        # Set the current column positions.
                        ctb = top_c + ctbdr * (float(rq)+0.5-top_r)
                        cmb = mid_c + cmbdr * (float(rq)+0.5-mid_r)

                        # The first pixel column for the current row
                        cq_start = max(int(np.floor(0.5 + min(cmb, ctb))), 0)
                        
                        # The last pixel column for the current row.
                        cq_end = min(int(np.floor(0.5 + max(cmb, ctb))), width)

                        for cq in range(cq_start, cq_end):
                            l1, l2, l3 = get_barycentric_coordinates(v1, v2, v3, float(cq) + 0.5, float(rq) + 0.5)
                            if l1 < 0 or l2 < 0 or l3 < 0:
                                continue

                            z = v1[2] * l1 + v2[2] * l2 + v3[2] * l3

                            depth_value = depth[cq,rq]
                            if z < depth_value:
                                # Current is behind another, don't draw
                                continue

                            # The uv values of the point
                            u = v1t[0] * l1 + v2t[0] * l2 + v3t[0] * l3
                            v = v1t[1] * l1 + v2t[1] * l2 + v3t[1] * l3

                            # Render pixel
                            s = u * float(cols)
                            t = (1.0 - v) * float(rows)

                            dl, left_f = np.modf(s)
                            dr = 1.0 - dl
                            dt, top_f = np.modf(t)
                            db = 1.0 - dt

                            left = int(left_f)
                            top = int(top_f)

                            for i in range(0, num_channels):
                                value = 0.0

                                tl = float(texture[i,top,left])
                                tr = float(texture[i,top,left+1])
                                bl = float(texture[i,top+1,left])
                                br = float(texture[i,top+1,left+1])

                                value += tl * dr * db
                                value += tr * dl * db
                                value += bl * dr * dt
                                value += br * dl * dt

                                bands[current_band_index + i,cq,rq] = value

                            # Increment the alpha band if the pixel was visible for this band
                            # the final alpha band will be set to 255 if alpha == num bands
                            # (all bands have information at this pixel)
                            alpha_band[cq,rq] += num_channels

                            # Update depth buffer
                            depth[cq,rq] = z
            _info("Material %s rendered" % mat)

            mat_idx += 1

        current_band_index += num_channels
        primary = False

    # TODO: write TIFF
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': len(bands),
        'dtype': data_type,
        # 'transform': rasterio.transform.Affine(dem_transform[0], dem_transform[1], offset_x, 
        #                                         dem_transform[3], dem_transform[4], offset_y),
        # 'nodata': None,
        # 'crs': crs
    }

    with rasterio.open(output, 'w', BIGTIFF="IF_SAFER", **profile) as wout:
        #for b in range(len(bands)):
        wout.write(bands)
        #wout.write(alpha_band, num_bands + 1)


set_renderer_logger(print, print)

render_orthophoto(["/datasets/brighton2/odm_texturing_25d/odm_textured_model_geo.obj"], 5.0, "/datasets/brighton2/odm_texturing_25d/test.tif")


# TODO - Compute area boundaries
# 