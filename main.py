import cv2
import numpy as np

from data.dataloader import TrainGenerator


def main():
    ds = TrainGenerator(
        "/home/ruslannuriev/Desktop/adapts/bottle/train/good",
        "/home/ruslannuriev/Desktop/adapts/dtd/images",
        (256, 256),
    )
    a = None
    for x in iter(ds):
        if x["has_anomaly"] == np.array([1.0], dtype=np.float32):
            a = x
        break
    cv2.imshow("Display Window", a["anomaly_image"])

    # 3. Keep the window open until any key is pressed
    cv2.waitKey(0)

    # 4. Clear the window from memory
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
