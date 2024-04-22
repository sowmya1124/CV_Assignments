import cv2

camera_id = 0
delay = 1
window_name = 'Object Tracking using QR Code'

qcd = cv2.QRCodeDetector()
cap = cv2.VideoCapture(camera_id)

while True:
    ret, frame = cap.read()

    if ret:
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
        if ret_qr:
            for s, p in zip(decoded_info, points):
                if s:
                    #print(s)
                    # Draw rectangle
                    frame = cv2.polylines(frame, [p.astype(int)], True, (0, 255, 0), 8)
                    # Add text
                    cv2.putText(frame, s, (int(p[0][0]), int(p[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    color = (0, 0, 255)
                    frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
        cv2.imshow(window_name, frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)
