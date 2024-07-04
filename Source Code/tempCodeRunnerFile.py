if (count < 60):
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