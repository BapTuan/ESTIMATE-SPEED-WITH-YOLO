import sys
import cv2
import numpy as np
import PySimpleGUI as sg
from ultralytics import YOLO
from collections import defaultdict, deque

import sys
sys.path.append('OC_SORT/trackers/ocsort_tracker')
from ocsort_custom import OCSort
print("OCSort being used from:", OCSort.__module__)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Khởi tạo ma trận biến đổi phối cảnh từ điểm nguồn đến điểm đích."""
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Chuyển đổi điểm từ hệ tọa độ gốc sang hệ tọa độ được biến đổi."""
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def calculate_speed(start_frame, end_frame, distance, fps):
    """Tính toán tốc độ từ thời gian và khoảng cách."""
    time_elapsed = (end_frame - start_frame) / fps
    return (distance / time_elapsed) * 3.6 if time_elapsed > 0 else None

# Layout của giao diện chính
layout = [
    [sg.Text('Video Path'), sg.InputText(), sg.FileBrowse(key='video_path')],
    [sg.Text('Model Path'), sg.InputText(), sg.FileBrowse(key='model_path')],
    [sg.Text('Output Path'), sg.InputText(), sg.FileSaveAs(key='output_path')],
    [sg.Text('Speed Threshold (km/h)'), sg.InputText('120', key='speed_thresh')],
    [sg.Button('Advanced Settings'), sg.Button('Start Processing'), sg.Button('Exit')],
    [sg.Image(filename='', key='-IMAGE-')]
]

# Layout của cài đặt nâng cao
advanced_layout = [
    [sg.Text('Source Points (comma separated, format: x y, x y, ...)')],
    [sg.InputText('980 300, 1630 300, 3130 1440, -500 1440', key='src_points')],
    [sg.Text('Destination Width')],
    [sg.InputText('25', key='dst_width')],
    [sg.Text('Destination Height')],
    [sg.InputText('230', key='dst_height')],
    [sg.Button('Save Settings'), sg.Button('Cancel')]
]

def create_polygon(src_points):
    """Tạo đa giác phân vùng từ điểm nguồn."""
    return np.array([src_points], dtype=np.int32)

def main():
    # Tạo cửa sổ chính
    window = sg.Window('Vehicle Speed Estimation', layout, finalize=True)
    advanced_window = None

    while True:
        event, values = window.read(timeout=10)

        # Thoát ứng dụng nếu đóng cửa sổ hoặc nhấn nút Exit
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        # Hiển thị cửa sổ cài đặt nâng cao
        if event == 'Advanced Settings':
            advanced_window = sg.Window('Advanced Settings', advanced_layout)

        if advanced_window:
            advanced_event, advanced_values = advanced_window.read(timeout=10)
            if advanced_event == sg.WIN_CLOSED or advanced_event == 'Cancel':
                advanced_window.close()
                advanced_window = None
            if advanced_event == 'Save Settings':
                # Chuyển đổi điểm nguồn và kích thước đích từ giao diện người dùng
                src_points = np.float32([
                    list(map(int, p.split())) for p in advanced_values['src_points'].split(',')
                ])
                dst_width = int(advanced_values['dst_width'])
                dst_height = int(advanced_values['dst_height'])

                # Tạo điểm đích và mặt nạ phân vùng
                dst_points = np.float32([
                    [0, 0],
                    [dst_width - 1, 0],
                    [dst_width - 1, dst_height - 1],
                    [0, dst_height - 1],
                ])
                polygon = create_polygon(src_points)
                advanced_window.close()
                advanced_window = None

        # Xử lý video khi nhấn nút Start Processing
        if event == 'Start Processing':
            video_path = values['video_path']
            model_path = values['model_path']
            output_path = values['output_path']

            if not output_path.endswith('.mp4'):
                output_path += '.mp4'

            speed_thresh = float(values['speed_thresh'])

            if not all([video_path, model_path, output_path]):
                sg.popup_error("Please specify all paths.")
                continue

            # Khởi tạo mô hình YOLO
            model = YOLO(model_path)

            # Mở video và kiểm tra
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                sg.popup_error("Error: Unable to open video.")
                continue

            # Đọc thông tin video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Khởi tạo VideoWriter để lưu video kết quả
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Khởi tạo OC-SORT tracker
            tracker = OCSort(det_thresh=0.6, max_age=60)

            # Danh sách các loại phương tiện cần thiết
            necessary_classes = {2, 3, 5, 7}

            # Bảng ánh xạ lớp (class) đến màu sắc
            class_colors = {
                2: (0, 255, 0),   # Màu xanh lá cho car
                3: (255, 255, 0), # Màu vàng cho truck
                5: (171, 130, 255), # Màu hồng cho bus
                7: (0, 255, 255), # Màu lục lam cho motorbike
                8: (192, 192, 192) # Màu xám cho motorbike
            }

            # Tạo mặt nạ dựa trên src_points (ROI)
            mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.fillPoly(mask, polygon, (255, 255, 255))

            # Tính toán ma trận biến đổi phối cảnh
            view_transformer = ViewTransformer(source=src_points, target=dst_points)

            # Khởi tạo cấu trúc lưu trữ tọa độ và khung hình
            coordinates = defaultdict(lambda: deque(maxlen=fps))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Áp dụng mặt nạ để chỉ giữ lại vùng quan tâm
                roi = cv2.bitwise_and(frame, mask)

                # Phát hiện đối tượng
                results = model(roi)

                # Tạo danh sách các detections từ YOLO
                detections = [
                    [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), box.conf[0]]
                    for result in results
                    for box in result.boxes
                    if int(box.cls) in necessary_classes
                ]

                # Cung cấp thông tin hình ảnh cho OC-SORT
                img_info = (frame.shape[1], frame.shape[0])
                img_size = (frame.shape[1], frame.shape[0])

                # Theo dõi các đối tượng
                tracked_objects = tracker.update(np.array(detections), img_info, img_size)

                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj.astype(int)
                    center_x, center_y = (x1 + x2) // 2, y2

                    # Chuyển đổi điểm trung tâm từ ảnh gốc sang ảnh đã biến đổi
                    original_point = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
                    transformed_point = view_transformer.transform_points(original_point)
                    transformed_x, transformed_y = transformed_point[0]

                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    coordinates[track_id].append((transformed_y, current_frame))

                    # Tính toán tốc độ
                    if len(coordinates[track_id]) > 1:
                        start_y, start_frame = coordinates[track_id][0]
                        end_y, end_frame = coordinates[track_id][-1]
                        distance = abs(start_y - end_y)
                        speed = calculate_speed(start_frame, end_frame, distance, fps)

                        if speed is not None:
                            cls = next(
                                (int(box.cls) for result in results for box in result.boxes
                                 if (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) == (x1, y1, x2, y2)),
                                -1
                            )

                            if cls != -1:
                                # Hiển thị bounding box và tốc độ
                                label_top = f'#{track_id}-{model.names[cls]}'
                                label_bottom = f'{int(speed)} km/h'
                                color = class_colors[cls] if speed < speed_thresh else (0, 0, 255)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, label_top, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                                cv2.putText(frame, label_bottom, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # Vẽ đa giác phân vùng lên frame
                cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=4)

                # Điều chỉnh kích thước khung hình
                scale_percent = 30
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)

                # Thay đổi kích thước khung hình
                resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                # Ghi frame đã chú thích vào tệp video
                out.write(frame)

                # Hiển thị frame đã chú thích
                cv2.imshow('Estimate Speed with YOLOv8', resized_frame)

                # Nhấn phím 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Giải phóng tài nguyên
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # Thông báo cho người dùng khi hoàn tất
            sg.popup('Processing Complete', 'The video has been processed and saved successfully!')

            # Đóng tất cả các cửa sổ sau khi hoàn tất
            break

    window.close()

if __name__ == "__main__":
    main()