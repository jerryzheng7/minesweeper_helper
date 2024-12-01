import cv2
import numpy as np
import pyautogui


def minesweeper(region, d_size):
    """Screen Capture Function"""
    try:
        while True:
            image = cv2.cvtColor(np.array(pyautogui.screenshot(region=region)), cv2.COLOR_RGB2BGR)
            image1 = cv2.resize(image, d_size)
            cv2.imshow("Minesweeper Helper", image1)
            if cv2.waitKey(0) == ord(' '):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    w,h = pyautogui.size()
    size = (800, 600)
    coordinates = ((w-size[0])//2,
                   (h-size[1])//2, *size,
    )
    minesweeper(region=coordinates, d_size=(640, 480))