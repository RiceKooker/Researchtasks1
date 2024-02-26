import pyautogui
import time

# 192.168.1.102 research laptop ip.
def active():
    start = time.time()
    while True:
        try:
            pyautogui.press('volumeup')
            pyautogui.press('volumedown')
            time.sleep(300)
        except (KeyboardInterrupt, SystemExit):
            end = time.time()
            print('%s %s' % (round((end - start) / 60), 'Minutes, Your Welcome :-)'))
            break


active()