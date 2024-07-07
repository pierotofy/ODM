import os
from PIL import Image
import numpy as np
from opendm import system
from opendm import log

from opendm.thermal_tools.thermal_utils import sensor_vals_to_temp
from opendm.exiftool import extract_temperature_params_from_image

def extract_temperatures_dji(photo, dataset_tree):
        """Extracts the DJI-encoded thermal image as 2D floating-point numpy array with temperatures in degC.
        The raw sensor values are obtained using the sample binaries provided in the official Thermal SDK by DJI.
        The executable file is run and generates a 16 bit unsigned RAW image with Little Endian byte order.
        Link to DJI Forum post: https://forum.dji.com/forum.php?mod=redirect&goto=findpost&ptid=230321&pid=2389016
        """
        meta = extract_temperature_params_from_image(os.path.join(dataset_tree, photo.filename))
        # if not meta:
        #     log.ODM_WARNING("Cannot extract temperature parameters from %s" % photo.filename)
        #     return None
        
        if photo.camera_model in ["MAVIC2-ENTERPRISE-ADVANCED", "M30T"]:
            im = Image.open(f"{dataset_tree}/{photo.filename}")
            # concatenate APP3 chunks of data
            a = im.applist[3][1]
            for i in range(4, 14):
                a += im.applist[i][1]
            # create image from bytes
            try:
                img = np.array(Image.frombytes("I;16L", (640, 512), a))
            except ValueError as e:
                log.ODM_ERROR("Error during extracting temperature values for file %s : %s" % photo.filename, e)

            if photo.camera_model == "M30T":
                # concatenate APP5 chunks of data
                a = im.applist[5][1]
                for i in range(6, 14):
                    a += im.applist[i][1]
                # create image from bytes
                print(a)
                print(len(a))
                try:
                    img = np.array(Image.frombytes("I;16L", (640, 512), a))
                except ValueError as e:
                    log.ODM_ERROR("Error during extracting temperature calibration values for file %s : %s" % photo.filename, e)
                exit(1)
        else:
            log.ODM_WARNING("Only DJI M2EA currently supported, please wait for new updates")
            return None

        return sensor_vals_to_temp(img, **meta)