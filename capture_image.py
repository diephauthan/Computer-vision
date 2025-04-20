import pypylon.pylon as pylon
import cv2
import sys
import os
import time

# Biến toàn cục để xử lý sự kiện chuột
capture_image = False

# Hàm xử lý sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    global capture_image
    if event == cv2.EVENT_RBUTTONDOWN:  # Kiểm tra sự kiện chuột phải
        capture_image = True

try:
    # Tạo thư mục để lưu ảnh nếu chưa tồn tại
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
        print("Đã tạo thư mục 'captured_images' để lưu ảnh")

    # Khởi tạo camera
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    
    if len(devices) == 0:
        print("Không tìm thấy camera. Vui lòng kiểm tra kết nối.")
        sys.exit(1)
    
    # Kết nối với camera đầu tiên
    camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    print(f"Đang kết nối với camera: {camera.GetDeviceInfo().GetModelName()}")
    camera.Open()
    
    # Cấu hình camera
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    # Tạo cửa sổ hiển thị
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)
    
    print("Đã mở camera, đang hiển thị hình ảnh.")
    print("Nhấn CHUỘT PHẢI để chụp và lưu ảnh, nhấn ESC để đóng.")
    
    # Hiển thị video
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grabResult.GrabSucceeded():
            # Chuyển đổi ảnh sang định dạng OpenCV
            image = converter.Convert(grabResult)
            img = image.GetArray()
            
            # Hiển thị ảnh
            cv2.imshow('Basler Camera', img)
            
            # Kiểm tra nếu chuột phải được nhấn để chụp ảnh
            if capture_image:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, img)
                print(f"Đã chụp và lưu ảnh tại: {filename}")
                capture_image = False
            
            # Kiểm tra phím ESC để thoát
            k = cv2.waitKey(1)
            if k == 27:  # ESC
                print("Đã nhấn ESC, đang đóng camera...")
                break
        
        grabResult.Release()
    
    # Giải phóng tài nguyên
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Đã đóng camera và kết thúc chương trình.")
    
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
    # Đảm bảo camera được đóng trong trường hợp có lỗi
    try:
        if 'camera' in locals():
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if camera.IsOpen():
                camera.Close()
            print("Đã đóng kết nối camera")
        cv2.destroyAllWindows()
    except:
        pass