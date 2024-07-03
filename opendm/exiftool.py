import json
import os
import tempfile
import base64
from rasterio.io import MemoryFile
from opendm.system import run
from opendm import log
from opendm.utils import double_quote

def extract_raw_thermal_image_data(image_path):
    try:
        f, tmp_file_path = tempfile.mkstemp(suffix='.json')
        os.close(f)

        try:
            output = run("exiftool -b -x ThumbnailImage -x PreviewImage -j \"%s\" > \"%s\"" % (image_path, tmp_file_path), quiet=True)

            with open(tmp_file_path) as f:
                j = json.loads(f.read())

                if isinstance(j, list):
                    j = j[0] # single file
                    
                    if "RawThermalImage" in j:
                        imageBytes = base64.b64decode(j["RawThermalImage"][len("base64:"):])

                        with MemoryFile(imageBytes) as memfile:
                            with memfile.open() as dataset:
                                img = dataset.read()
                                bands, h, w = img.shape

                                if bands != 1:
                                    raise Exception("Raw thermal image has more than one band? This is not supported")

                                # (1, 512, 640) --> (512, 640, 1)
                                img = img[0][:,:,None]

                        del j["RawThermalImage"]
                    
                    return extract_temperature_params_from(j), img
                else:
                    raise Exception("Invalid JSON (not a list)")

        except Exception as e:
            log.ODM_WARNING("Cannot extract tags using exiftool: %s" % str(e))
            return {}, None
        finally:
            if os.path.isfile(tmp_file_path):
                os.remove(tmp_file_path)
    except Exception as e:
        log.ODM_WARNING("Cannot create temporary file: %s" % str(e))
        return {}, None

def unit(unit):
    def _convert(v):
        if isinstance(v, float):
            return v
        elif isinstance(v, str):
            if not v[-1].isnumeric():
                if v[-1].upper() != unit.upper():
                    log.ODM_WARNING("Assuming %s is in %s" % (v, unit))
                return float(v[:-1])
            else:
                return float(v)
        else:
            return float(v)
    return _convert

    

def extract_temperature_params_from(tags):
    # Defaults

    meta = {
        "Emissivity": (float, 0.95),
        "ObjectDistance": (unit("m"), 50),
        "AtmosphericTemperature": (unit("C"), 20),
        "ReflectedApparentTemperature": (unit("C"), 30),
        "IRWindowTemperature": (unit("C"), 20),
        "IRWindowTransmission": (float, 1),
        "RelativeHumidity": (unit("%"), 40),
        "PlanckR1": (float, 21106.77),
        "PlanckB": (float, 1501),
        "PlanckF": (float, 1),
        "PlanckO": (float, -7340),
        "PlanckR2": (float, 0.012545258),
    }

    aliases = {
        "AtmosphericTemperature": ["AmbientTemperature"],
        "ReflectedApparentTemperature": ["ReflectedTemperature"],
    }

    params = {}

    for m in meta:
        keys = [m]
        keys += aliases.get(m, [])
        val = None
        for k in keys:
            if k in tags:
                unit_func = meta[m][0]
                val = unit_func(tags[k])
                break
        if val is None:
            val = meta[m][1] # Default
        if val is None:
            raise Exception("Cannot find %s in tags" % m)
        
        params[m] = val
    
    # params["PlanckR1"] = 14364.633
    # params["PlanckO"] = -8192
    return params

    
def extract_temperature_params_from_image(image_path):
    try:
        f, tmp_file_path = tempfile.mkstemp(suffix='.json')
        os.close(f)

        try:
            output = run("exiftool -b -x ThumbnailImage -x PreviewImage -j \"%s\" > \"%s\"" % (image_path, tmp_file_path), quiet=True)

            with open(tmp_file_path) as f:
                j = json.loads(f.read())

                if isinstance(j, list):
                    j = j[0] # single file
                    
                    return extract_temperature_params_from(j)
                else:
                    raise Exception("Invalid JSON (not a list)")

        except Exception as e:
            log.ODM_WARNING("Cannot extract temperature params using exiftool: %s" % str(e))
            return {}
        finally:
            if os.path.isfile(tmp_file_path):
                os.remove(tmp_file_path)
    except Exception as e:
        log.ODM_WARNING("Cannot create temporary file: %s" % str(e))
        return {}