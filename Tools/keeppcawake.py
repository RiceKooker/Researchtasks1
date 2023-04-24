import pyautogui
import time


def active():
    start = time.time()
    while True:
        try:
            pyautogui.press('volumedown')
            time.sleep(100)
            pyautogui.press('volumeup')
            time.sleep(300)
        except (KeyboardInterrupt, SystemExit):
            end = time.time()
            print('%s %s' % (round((end - start) / 60), 'Minutes, Your Welcome :-)'))
            break


active()