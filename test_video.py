import cv2
from fer import FER

detector = FER(mtcnn=True)

video = cv2.VideoCapture("videos/test.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = detector.detect_emotions(frame)

    for face in results:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, top_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Video Emotion Detection", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
