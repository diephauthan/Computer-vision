import numpy as np
import cv2
import glob
import os
import pickle

chessboard_size = (9, 6)  # 9 góc theo chiều ngang và 6 góc theo chiều dọc
frameSize = (640, 480)
square_size = 24

# Các hàm hỗ trợ (giữ nguyên)
def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# Set termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_points = []  # Store vectors of 3D points for all chessboard images (world coordinate frame)
img_points = []  # Store vectors of 2D points for all chessboard images (camera coordinate frame)
image_names = []  # Store image filenames to match with calibration matrices later

# Tạo lưới các điểm 3D cho bàn cờ
obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_p *= square_size  # Quy đổi kích thước các điểm cho đúng với thực tế

# Tạo thư mục đầu ra cho ảnh hiệu chuẩn
output_dir = 'calibration_results'
camera_calib_dir = os.path.join(output_dir, 'camera_calibration')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(camera_calib_dir):
    os.makedirs(camera_calib_dir)

# Đọc tất cả các ảnh trong thư mục
input_dir = '/home/thandiep/test/captured_images'
images = glob.glob(f'{input_dir}/*.jpg')  # Tìm tất cả file jpg trong thư mục

if len(images) == 0:
    print(f"Không tìm thấy ảnh trong thư mục '{input_dir}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

print(f"Đã tìm thấy {len(images)} ảnh. Bắt đầu xử lý...")

# Tạo cửa sổ với kích thước cố định cho hiển thị bàn cờ
cv2.namedWindow('Chessboard Corners', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Chessboard Corners', 800, 600)  # Kích thước cửa sổ 800x600

successful_images = 0
for i, image_path in enumerate(images):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh {image_path}. Bỏ qua.")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tìm các góc của bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    if ret:
        successful_images += 1
        # Tinh chỉnh vị trí các góc với độ chính xác subpixel
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        obj_points.append(obj_p)
        img_points.append(corners2)
        
        # Lưu tên file ảnh để mapping sau này
        base_filename = os.path.basename(image_path)
        image_names.append(base_filename)

        # Vẽ các góc trên ảnh để kiểm tra
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        
        # Lưu ảnh có đánh dấu góc
        calibration_img_path = os.path.join(output_dir, f'calibration_{base_filename}')
        cv2.imwrite(calibration_img_path, img)
        
        # Thay đổi kích thước ảnh cho hiển thị
        display_img = resize_image(img, width=800)
        
        cv2.imshow('Chessboard Corners', display_img)
        key = cv2.waitKey(1000)
        if key == 27:
            break
    else:
        print(f"Không phát hiện được bàn cờ trong ảnh {image_path}")

cv2.destroyAllWindows()

print(f"Phát hiện thành công bàn cờ trong {successful_images}/{len(images)} ảnh")

if successful_images < 5:
    print("Cảnh báo: Số lượng ảnh hiệu chuẩn quá ít, kết quả có thể không chính xác.")

if len(obj_points) > 0:
    print("Đang hiệu chuẩn camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, frameSize, None, None)
  
    #reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(obj_points)
    print(f"Mean reprojection error: {mean_error}")

    print("Calibration camera results:")
    print(f"Intrinsic Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist.ravel()}")
    
    # ===== CẢI TIẾN: Lưu các ma trận kèm theo tên file ảnh =====
    # Phương pháp 1: Lưu dạng npz cơ bản với array riêng cho tên file
    np.savez(os.path.join(camera_calib_dir, "camera_calibration_with_images.npz"), 
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, 
             image_names=np.array(image_names, dtype=object), error=mean_error)
    
    # Phương pháp 2: Lưu dạng dictionary cho dễ truy xuất
    calib_data_dict = {}
    for i, img_name in enumerate(image_names):
        calib_data_dict[img_name] = {
            'rvec': rvecs[i],
            'tvec': tvecs[i]
        }
    
    # Lưu riêng rvecs và tvecs như cũ để tương thích với code cũ
    np.save(os.path.join(camera_calib_dir, "rvecs.npy"), rvecs)
    np.save(os.path.join(camera_calib_dir, "tvecs.npy"), tvecs)
    
    # Lưu dictionary các ma trận dưới dạng pickle
    with open(os.path.join(camera_calib_dir, "calibration_matrices_dict.pkl"), "wb") as f:
        pickle.dump(calib_data_dict, f)
    
    # Lưu dictionary dưới dạng npz để dễ load
    np.savez(os.path.join(camera_calib_dir, "calibration_matrices_dict.npz"), 
             calib_dict=calib_data_dict)
    
    # Lưu danh sách tên files để map riêng
    np.save(os.path.join(camera_calib_dir, "image_names.npy"), np.array(image_names, dtype=object))
    
    # Lưu kết quả dạng văn bản để dễ đọc
    with open(os.path.join(output_dir, "calibration_results.txt"), "w") as f:
        f.write(f"Camera Matrix:\n{mtx}\n\n")
        f.write(f"Distortion Coefficients:\n{dist.ravel()}\n\n")
        f.write(f"Reprojection Error: {mean_error}\n\n")
        f.write("Image to matrix mapping:\n")
        for i, img_name in enumerate(image_names):
            f.write(f"{i}: {img_name}\n")
    
    # Lưu các thông số khác như cũ
    pickle.dump((mtx, dist), open(os.path.join(output_dir, "calibration.pkl"), "wb"))
    pickle.dump(mtx, open(os.path.join(output_dir, "cameraMatrix.pkl"), "wb"))
    pickle.dump(dist, open(os.path.join(output_dir, "dist.pkl"), "wb"))

# ============================================================= #
# Code để tạo ma trận extrinsic từ rvecs và tvecs cùng tên file #
# ============================================================= #

def create_extrinsic_matrices():
    # Load dữ liệu calibration
    try:
        data = np.load(os.path.join(camera_calib_dir, "camera_calibration_with_images.npz"), allow_pickle=True)
        rvecs = data['rvecs']
        tvecs = data['tvecs']
        image_names = data['image_names']
        
        # Dictionary lưu ma trận extrinsic
        extrinsic_dict = {}
        
        # Tạo ma trận extrinsic cho mỗi cặp rvec và tvec
        for i, (rvec, tvec, img_name) in enumerate(zip(rvecs, tvecs, image_names)):
            # Chuyển đổi rotation vector thành rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Tạo ma trận extrinsic 4x4 [R|t; 0 0 0 1]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = tvec.flatten()
            
            # Lưu vào dictionary
            extrinsic_dict[img_name] = extrinsic
        
        # Lưu dictionary extrinsic
        with open(os.path.join(camera_calib_dir, "extrinsic_matrices.pkl"), "wb") as f:
            pickle.dump(extrinsic_dict, f)
            
        # Lưu dưới dạng npz
        np.savez(os.path.join(camera_calib_dir, "extrinsic_matrices.npz"), 
                 extrinsic_dict=extrinsic_dict)
        
        print(f"Đã tạo và lưu ma trận extrinsic cho {len(extrinsic_dict)} ảnh")
        
        # In ra ma trận đầu tiên để kiểm tra
        example_img = image_names[0]
        print(f"\nMa trận extrinsic cho ảnh {example_img}:")
        print(extrinsic_dict[example_img])
        
        return extrinsic_dict
        
    except Exception as e:
        print(f"Lỗi khi tạo ma trận extrinsic: {e}")
        return None

# Để sử dụng hàm này sau quá trình calibration:
extrinsic_dict = create_extrinsic_matrices()

# ==== Code to load và use dữ liệu calibration ====

def load_calibration_data():
    """Load dữ liệu calibration và mapping giữa ảnh và ma trận"""
    try:
        # Cách 1: Load từ file npz có tên ảnh
        data = np.load(os.path.join(camera_calib_dir, "camera_calibration_with_images.npz"), 
                      allow_pickle=True)
        mtx = data['mtx']
        dist = data['dist']
        rvecs = data['rvecs']
        tvecs = data['tvecs']
        image_names = data['image_names']
        
        # Tạo dictionary map giữa tên ảnh và index
        image_map = {name: i for i, name in enumerate(image_names)}
        
        print(f"Đã load dữ liệu calibration cho {len(image_names)} ảnh")
        return mtx, dist, rvecs, tvecs, image_names, image_map
        
    except Exception as e:
        print(f"Lỗi khi load dữ liệu calibration: {e}")
        return None, None, None, None, None, None

def load_extrinsic_dict():
    """Load dictionary các ma trận extrinsic"""
    try:
        # Cách 1: Load từ file pickle
        with open(os.path.join(camera_calib_dir, "extrinsic_matrices.pkl"), "rb") as f:
            extrinsic_dict = pickle.load(f)
            
        # Hoặc Cách 2: Load từ file npz
        # data = np.load(os.path.join(camera_calib_dir, "extrinsic_matrices.npz"), allow_pickle=True)
        # extrinsic_dict = data['extrinsic_dict'].item()
        
        print(f"Đã load {len(extrinsic_dict)} ma trận extrinsic")
        return extrinsic_dict
        
    except Exception as e:
        print(f"Lỗi khi load ma trận extrinsic: {e}")
        return None

def load_calibration_matrices_dict():
    """Load dictionary các ma trận rvec và tvec"""
    try:
        # Cách 1: Load từ file pickle
        with open(os.path.join(camera_calib_dir, "calibration_matrices_dict.pkl"), "rb") as f:
            calib_dict = pickle.load(f)
            
        # Hoặc Cách 2: Load từ file npz
        # data = np.load(os.path.join(camera_calib_dir, "calibration_matrices_dict.npz"), allow_pickle=True)
        # calib_dict = data['calib_dict'].item()
        
        print(f"Đã load dữ liệu calibration cho {len(calib_dict)} ảnh")
        return calib_dict
        
    except Exception as e:
        print(f"Lỗi khi load ma trận calibration: {e}")
        return None
    
def get_extrinsic_for_image(image_filename):
    """Lấy ma trận extrinsic cho một ảnh khi biết tên file"""
    try:
        # Load extrinsic dict
        extrinsic_dict = load_extrinsic_dict()
        if extrinsic_dict is None:
            return None
        
        # Lấy tên file base (không có đường dẫn)
        base_filename = os.path.basename(image_filename)
        
        # Kiểm tra xem ảnh có trong dictionary không
        if base_filename in extrinsic_dict:
            return extrinsic_dict[base_filename]
        else:
            print(f"Không tìm thấy ma trận extrinsic cho ảnh {base_filename}")
            print(f"Danh sách các ảnh có sẵn: {list(extrinsic_dict.keys())}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy ma trận extrinsic: {e}")
        return None
    
# Hàm cho phép người dùng nhập tên ảnh và trả về ma trận
def find_extrinsic_by_image_name():
    """Hàm tương tác cho phép nhập tên ảnh và hiển thị ma trận extrinsic"""
    print("\n=== TÌM MA TRẬN EXTRINSIC THEO TÊN ẢNH ===")
    
    # Hiển thị danh sách các ảnh có sẵn
    try:
        extrinsic_dict = load_extrinsic_dict()
        if extrinsic_dict is None:
            print("Không thể load dữ liệu ma trận extrinsic")
            return
            
        # Hiển thị danh sách các ảnh
        print("Danh sách ảnh có sẵn:")
        for i, img_name in enumerate(extrinsic_dict.keys()):
            print(f"{i+1}. {img_name}")
        
        # Nhập tên ảnh
        image_name = input("\nNhập tên file ảnh (hoặc đường dẫn đầy đủ): ")
        
        # Tìm và hiển thị ma trận
        extrinsic = get_extrinsic_for_image(image_name)
        if extrinsic is not None:
            print(f"\nMa trận extrinsic cho ảnh {os.path.basename(image_name)}:")
            print(extrinsic)
    except Exception as e:
        print(f"Lỗi: {e}")

# Ví dụ sử dụng:
if __name__ == "__main__":
    # 1. Tạo extrinsic matrices nếu chưa có
    if not os.path.exists(os.path.join(camera_calib_dir, "extrinsic_matrices.pkl")):
        print("Tạo ma trận extrinsic...")
        extrinsic_dict = create_extrinsic_matrices()
    
    # 2. Gọi hàm tìm ma trận theo tên
    find_extrinsic_by_image_name()
    
    # 3. Hoặc sử dụng trực tiếp:
    # image_name = "image_20250417_193744.jpg"  # Thay bằng tên ảnh cần tìm
    # extrinsic = get_extrinsic_for_image(image_name)
    # if extrinsic is not None:
    #     print(f"\nMa trận extrinsic cho ảnh {image_name}:")
    #     print(extrinsic)