# ==================================================
#                  Task Parameters
# ==================================================

import os
from ament_index_python.packages import get_package_share_directory

LOCATIONS = [(-6,2,1.7),(0,-2.5,1.7),(5,2.5,1.7)]

# Flight related
TAKEOFF_ALTITUDE = 2.5

# AI Model path 
try:
    package_share_directory = get_package_share_directory('faulty_or_not')
    MODEL_PATH = os.path.join(package_share_directory, 'models', 'best.pt')
except:
    # Fallback
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(current_dir)
    MODEL_PATH = os.path.join(package_root, 'models', 'best.pt')

