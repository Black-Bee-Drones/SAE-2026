# ==================================================
#                  Task Parameters
# ==================================================

import os
from ament_index_python.packages import get_package_share_directory

LOCATIONS = [(-5,-1.5,-6.5),(0,1.5,-6.5),(4,-2,-6.5)]

# Flight related
TAKEOFF_ALTITUDE = 2.5

# AI Model path 
try:
    package_share_directory = get_package_share_directory('faulty_or_not')
    MODEL_PATH = os.path.join(package_share_directory, 'models', 'best.pt')
    COARSE_MODEL_PATH = os.path.join(package_share_directory, 'models', 'coarse.pt')
except:
    # Fallback
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(current_dir)
    MODEL_PATH = os.path.join(package_root, 'models', 'best.pt')
    COARSE_MODEL_PATH = os.path.join(package_share_directory, 'models', 'coarse.pt')


# Camera details

DEVICE=0
GENERAL_EXPOSURE=1
FINE_EXPOSURE=1