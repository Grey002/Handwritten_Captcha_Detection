import cv2
import numpy as np
cam = cv2.VideoCapture(0)
count = 1
upper_left = (70, 70)
bottom_right = (320, 320)
background = None
accumulated_weight = 0.5


def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:

        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        return (thresholded, hand_segment_max_cont)


def mask_transform(img):
    min = np.array([0, 133, 77], np.uint8)
    max = np.array([235, 173, 127], np.uint8)
    ycr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    msk = cv2.inRange(ycr_img, min, max)
    skin = cv2.bitwise_and(img, img, mask=msk)
    return skin


def transform(img):
    h, w = img.shape[:2]
    new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(new, 150, 255, cv2.THRESH_BINARY_INV)
    cont, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cont]
    max_ind = np.argmax(areas)
    cnt = cont[max_ind]
    msk = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(msk, cont, max_ind, (255, 255, 255), cv2.FILLED)
    return msk


while (True):
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rect = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    gray_img = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

    # if (count < 60):
    #     cal_accum_avg(gray_img, accumulated_weight)
    #     cv2.putText(frame, "FETCHING BACKGROUND...PLEASE WAIT",
    #                 (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # else:
    #     hand = segment_hand(gray_frame)

    #     # Checking if we are able to detect the hand...
    #     if hand is not None:

    #         # unpack the thresholded img and the max_contour...
    #         thresholded, hand_segment = hand
    #         # Drawing contours around hand segment
    #         cv2.drawContours(frame, [hand_segment + (bottom_right[0],
    #                                                  upper_left[0])], -1, (255, 0, 0), 1)

    #         cv2.putText(frame_copy, str(num_frames), (70, 45),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #         cv2.putText(frame_copy, str(num_imgs_taken) + 'images' + "For"
    #                     + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                     (0, 0, 255), 2)

    #         # Displaying the thresholded image
    #         cv2.imshow("Thresholded Hand Image", thresholded)
    #         if num_imgs_taken <= 300:
    #             cv2.imwrite(r"D:\\gesture\\train\\"+str(element)+"\\" +
    #                         str(num_imgs_taken+300) + '.jpg', thresholded)

    #         else:
    #             break
    #         num_imgs_taken += 1
    #     else:
    #         cv2.putText(frame_copy, 'No hand detected...', (200, 400),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # rect = transform(rect)
    rect = mask_transform(rect)
    # n_rect = cv2.cvtColor(rect,cv2.COLOR_GRAY2RGB)
    frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = rect
    # cv2.imwrite('DATA/'+'img' + str(count)+'.jpeg',n_rect)
    cv2.imshow('frame', frame)
    if (cv2.waitKey(100) & 0xFF == ord('q')):
        break
    count += 1
cam.release()
cv2.destroyAllWindows()
