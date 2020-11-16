import cv2
import numpy as np

cam = cv2.VideoCapture('video.mp4')

scale_percent = 35
threshold = int(100)
sobel_vertical = np.float32([[-1, -2, -1],
                             [0, 0, 0],
                             [+1, +2, +1]])
sobel_horizontal = np.transpose(sobel_vertical)

left_top_y = 0
left_top_x = 0
left_bottom_y = 0
left_bottom_x = 0

while True:
    ret, frame = cam.read()
    if ret is False:
        break
    width = int(frame.shape[1] * scale_percent / 100)
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('2. Original frame', frame)
    cv2.imshow('3. Gray', gray)

    blank_video = np.zeros((height, width), dtype=np.uint8)
    upper_left = (int(width * 0.44), int(height * 0.80))
    upper_right = (int(width * 0.55), int(height * 0.80))
    lower_left = (0, height)
    lower_right = (width, height)
    trapezoid_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    cv2.fillConvexPoly(blank_video, trapezoid_points, (1, 1, 1))
    cv2.imshow('4. Only the road', blank_video * gray)

    frame_points = np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.float32(trapezoid_points), frame_points)
    warped = cv2.warpPerspective(gray, M, (width, height))
    blured = cv2.blur(warped, ksize=(3, 3))
    cv2.imshow('5. Top down view', warped)
    cv2.imshow('6. Blured', blured)

    filter_horizontal = cv2.filter2D(np.float32(warped), -1, sobel_horizontal)
    filter_vertical = cv2.filter2D(np.float32(blured), -1, sobel_vertical)
    binarize_frame = np.sqrt(filter_horizontal ** 2 + filter_vertical ** 2)
    cv2.imshow('7.Edge detection', cv2.convertScaleAbs(binarize_frame))

    binarize_frame = (binarize_frame < threshold) * 0 + (binarize_frame >= threshold) * 255
    cv2.imshow('8.Binarize', cv2.convertScaleAbs(binarize_frame))

    #9
    final_copy = binarize_frame.copy()
    final_copy[0:, 0:int(width * 0.05)] = 0
    final_copy[0:, int(width * 0.95):int(width)] = 0
    left_side_screen = final_copy[0:, 0:int(width * 0.45)]
    right_side_screen = final_copy[0:, int(width * 0.55):int(width)]
    left_white = np.argwhere(left_side_screen == 255)
    right_white = np.argwhere(right_side_screen == 255)
    left_xs = np.array(left_white[:, 1])
    left_ys = np.array(left_white[:, 0])
    right_xs = np.array(right_white[:, 1])
    right_ys = np.array(right_white[:, 0])

    black_frame = np.zeros((height, width), dtype=np.uint8)
    if len(left_xs) > 0 and len(left_ys) > 0:
        left_line = np.polyfit(left_xs, left_ys, deg=1)
        if -10 ** 8 <= -left_line[1] / left_line[0] <= 10 ** 8 and -10 ** 8 <= (height - left_line[1]) / left_line[
            0] <= 10 ** 8:
            left_top_x = int(0 - left_line[1] / left_line[0])
            left_top_y = 0
            left_bottom_y = int(height)
            left_bottom_x = int((height - left_line[1]) / left_line[0])
        cv2.line(binarize_frame, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (200, 0, 0), 6)
        cv2.line(black_frame, (left_top_x, left_top_y), (left_bottom_x, left_bottom_y), (255, 255, 255), 6)
    detected_frame = frame.copy()
    M = cv2.getPerspectiveTransform(frame_points, np.float32(trapezoid_points))
    warped1 = cv2.warpPerspective(black_frame, M, (width, height))
    left_line = np.argwhere(warped1 == 255)
    for p in left_line:
        detected_frame[p[0], p[1]] = [255, 0, 0]

    black_frame = np.zeros((height, width), dtype=np.uint8)
    if len(right_xs) > 0 and len(right_ys) > 0:
        right_line = np.polyfit(right_xs, right_ys, deg=1)
        if -10 ** 8 <= -right_line[1] / right_line[0] <= 10 ** 8 and -10 ** 8 <= (height - right_line[1]) / right_line[
            0] <= 10 ** 8:
            right_top_x = int(width * 0.55) + int(0 - right_line[1] / right_line[0])
            right_top_y = 0
            right_bottom_y = int(height)
            right_bottom_x = int(width * 0.55) + int((height - right_line[1]) / right_line[0])
        cv2.line(binarize_frame, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (200, 0, 0), 6)
        cv2.line(black_frame, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), (255, 255, 255), 6)

    M = cv2.getPerspectiveTransform(frame_points, np.float32(trapezoid_points))
    warped1 = cv2.warpPerspective(black_frame, M, (width, height))
    right_line = np.argwhere(warped1 == 255)
    for p in right_line:
        detected_frame[p[0], p[1]] = [0, 255, 0]

    cv2.imshow('10. Street marking', cv2.convertScaleAbs(binarize_frame))
    cv2.imshow('11. Line detection', detected_frame)

    # 11
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
