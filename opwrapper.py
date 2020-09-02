
import numpy as np
import sys

try:
    try:
        sys.path.append('openpose/build/python');
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "openpose/models"
    params["hand"] = False
    params["logging_level"] = 5
    params["net_resolution"] = "656x368"

except Exception as e:
    print(e)
    sys.exit(-1)

# From Python
# It requires OpenCV installed for Python
def get_keypoints(image):
    
    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        datum.cvInputData = image

        opWrapper.emplaceAndPop([datum])

        if(datum.poseKeypoints.ndim == 0):
            print("No person found in image!")
            sys.exit(-1)
        num = len(datum.poseKeypoints)
        if(num is 0):
            print("No person found in image!")
            sys.exit(-1)
        return datum

    except Exception as e:
        print("Openpose Error: ")
        print(e)




