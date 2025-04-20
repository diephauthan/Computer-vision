import numpy as np

def pixel_to_world_point(pixel_x, pixel_y, camera_matrix, pose_matrix):
    """
    Convert pixel coordinates to real world coordinates.
    
    Args:
        pixel_x (float): x-coordinate of the pixel
        pixel_y (float): y-coordinate of the pixel
        camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix
        pose_matrix (numpy.ndarray): 4x4 extrinsic camera pose matrix
        
    Returns:
        numpy.ndarray: 3D real world point
    """
    # Extract rotation matrix (3x3) from pose matrix
    rotation_matrix = pose_matrix[0:3, 0:3]
    
    # Extract translation vector from pose matrix
    translation_vector = pose_matrix[0:3, 3]
    
    # Get z value (third element of translation vector)
    z = translation_vector[2]
    
    # Create homogeneous pixel coordinates
    mp = np.array([pixel_x, pixel_y, 1])
    
    # Calculate inverse of camera matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    
    # Calculate inverse of rotation matrix
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    
    # Calculate: inverse_camera_matrix * z * mp
    term1 = np.dot(camera_matrix_inv, z * mp)
    
    # Calculate: term1 - translation_vector
    term2 = term1 - translation_vector
    
    # Calculate: inverse_rotation_matrix * term2
    real_point = np.dot(rotation_matrix_inv, term2)
    
    return real_point

# Define the camera matrix from the given data
camera_matrix = np.array([
    [1.51014145e+04, 0.00000000e+00, 1.72431288e+03],
    [0.00000000e+00, 1.49430631e+04, 1.46680985e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Define the pose matrix from the given data
pose_matrix = np.array([
    [9.82287379e-01, 3.40354934e-02, 1.84263641e-01, -8.22778242e+01],
    [-3.28819897e-02, 9.99415851e-01, -9.31300411e-03, -6.18875333e+01],
    [-1.84472976e-01, 3.08909125e-03, 9.82832732e-01, 1.08845620e+03],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# Example usage
if __name__ == "__main__":
    # Example: convert pixel coordinates (500, 400) to real world point
    pixel_x = 0
    pixel_y = 2748
    
    real_point = pixel_to_world_point(pixel_x, pixel_y, camera_matrix, pose_matrix)
    
    print(f"Pixel coordinates: ({pixel_x}, {pixel_y})")
    print(f"Real world coordinates (X, Y, Z): {real_point}")