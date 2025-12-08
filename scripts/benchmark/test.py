import os
import sys
import torch

# Add root directory to sys.path to find src
# CVAE/scripts/benchmark/test.py -> CVAE/
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

# Using the torch version as it was used in the previous file
from src.dynamics.urdf2robot_torch import urdf2robot

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Path to a1.urdf
    # Located at CVAE/assets/a1_description/urdf/a1.urdf relative to project root
    urdf_path = os.path.join(ROOT_DIR, "assets/a1_description/urdf/a1.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return

    print(f"Loading URDF from: {urdf_path}")

    try:
        # urdf2robot returns (robot, robot_keys)
        robot, robot_keys = urdf2robot(urdf_path, verbose_flag=True, device=device)
        
        print("\n=== Success! Robot Model Loaded ===")
        print(f"Robot Name: {robot['name']}")
        print(f"n_q: {robot['n_q']}")
        print(f"n_links_joints: {robot['n_links_joints']}")
        
        print("\nLinks:")
        for link in robot['links']:
            print(f"  ID: {link['id']}, Mass: {link['mass']}")
            
        print("\nJoints:")
        for joint in robot['joints']:
            print(f"  ID: {joint['id']}, Type: {joint['type']}, Axis: {joint['axis']}")

        print("\n=== Generalized Coordinates (q) Mapping ===")
        # robot_keys['q_id'] maps joint_name -> q_index
        q_map = robot_keys['q_id']
        # Sort by q_index
        sorted_q = sorted(q_map.items(), key=lambda item: item[1])
        
        for name, q_idx in sorted_q:
            print(f"q[{q_idx}]: {name}")

    except Exception as e:
        print(f"Failed to load URDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
