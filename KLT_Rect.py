import pypylon.pylon as pylon
import cv2
import sys
import os
import time
import numpy as np

# Biến toàn cục để xử lý sự kiện chuột
capture_image = False

# Hàm xử lý sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    global capture_image
    if event == cv2.EVENT_RBUTTONDOWN:  # Kiểm tra sự kiện chuột phải
        capture_image = True

# Khởi tạo KLT tracker
class KLTTracker:
    def __init__(self):
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.tracks = []
        self.prev_gray = None

    def init_tracker(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Phát hiện các điểm đặc trưng (features) trong ảnh
        points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if points is not None:
            # Lưu trữ các điểm đặc trưng dưới dạng mảng numpy đơn giản, không phải mảng lồng nhau
            self.tracks = [[(point[0][0], point[0][1])] for point in points]
        self.prev_gray = gray

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.tracks) > 0:
            prev_points = np.array([track[-1] for track in self.tracks], dtype=np.float32).reshape(-1, 1, 2)

            # Tính toán optical flow với các điểm đặc trưng
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_points, None, **self.lk_params)

            # Kiểm tra nếu new_points là mảng hợp lệ
            if new_points is None or new_points.shape[0] == 0:
                print("Lỗi: new_points không hợp lệ!")
                return frame

            new_tracks = []
            for i, (track, status_flag) in enumerate(zip(self.tracks, status.flatten())):
                if status_flag:
                    x = float(new_points[i][0][0])
                    y = float(new_points[i][0][1])
                    track.append((x, y))
                    new_tracks.append(track)

            self.tracks = new_tracks

        self.prev_gray = gray
        return frame

# Mở camera Basler
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()

if len(devices) == 0:
    print("Không tìm thấy camera. Vui lòng kiểm tra kết nối.")
    sys.exit(1)

camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
camera.Open()

# Cấu hình camera
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Khởi tạo KLT tracker
klt_tracker = KLTTracker()

cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Basler Camera', 800, 600)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = image.GetArray()

        # Cập nhật các điểm đặc trưng (KLT)
        if klt_tracker.prev_gray is None:
            klt_tracker.init_tracker(frame)
        else:
            frame = klt_tracker.update(frame)

        # Vẽ các điểm đặc trưng
        for track in klt_tracker.tracks:
            for point in track:
                cv2.circle(frame, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)

        # Vẽ hình chữ nhật bao quanh các điểm đặc trưng
        if len(klt_tracker.tracks) > 0:
            points = np.array([track[-1] for track in klt_tracker.tracks])

            # Kiểm tra nếu points không rỗng và chuyển đổi kiểu dữ liệu
            if points.size > 0:
                points = points.astype(np.float32)  # Chuyển đổi kiểu dữ liệu thành float32
                x, y, w, h = cv2.boundingRect(points)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                print("Không có điểm hợp lệ để vẽ hình chữ nhật.")


        cv2.imshow('Basler Camera', frame)

        # Kiểm tra nếu chuột phải được nhấn để chụp ảnh
        if capture_image:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captured_images/image_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Đã chụp và lưu ảnh tại: {filename}")
            capture_image = False

        # Kiểm tra phím ESC để thoát
        k = cv2.waitKey(1)
        if k == 27:  # ESC
            print("Đã nhấn ESC, đang đóng camera...")
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
print("Đã đóng camera và kết thúc chương trình.")
