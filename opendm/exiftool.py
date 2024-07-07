import json
import os
import tempfile
import base64
import numpy as np
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
                    
                    elif "ThermalData" in j:
                        thermal_data = base64.b64decode(j["ThermalData"][len("base64:"):])
                        thermal_data_buf = np.frombuffer(thermal_data, dtype=np.int16)

                        thermal_calibration = base64.b64decode(j["ThermalCalibration"][len("base64:"):])
                        thermal_calibration_buf = np.frombuffer(thermal_calibration, dtype=np.int16)
                        thermal_calibration_buf = thermal_calibration_buf[0x100:0x1d00]
                        # TODO: how to interpret these?
                        # https://exiftool.org/forum/index.php?topic=11401.45
                        # print(thermal_data_buf.shape)
                        # print(thermal_calibration_buf.shape)
                        # print(" ".join("%02x" % b for b in thermal_data_buf[0:10]))
                        # print(thermal_data_buf[0:10])
                        print(thermal_calibration_buf[128:128+40])
                        # print(thermal_calibration_buf[0x100:0x100 + 128])
                        # print(np.min(thermal_data_buf))
                        # print(np.max(thermal_data_buf))
                        # print(np.min(thermal_calibration_buf))
                        # print(np.max(thermal_calibration_buf))
                        

                        with open("/datasets/dji_thermal/calib.raw", "wb") as f:
                            f.write(thermal_calibration)
                        exit(1)
                    
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
        "PlanckR1": (float, None),
        "PlanckB": (float, None),
        "PlanckF": (float, None),
        "PlanckO": (float, None),
        "PlanckR2": (float, None),
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
            val = meta[m][1] # Default, if available
        if val is None:
            raise Exception("Cannot find %s in tags" % m)
        
        params[m] = val
    
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