import cv2
import random 
import numpy as np
import utils.img_utils as img_utils

def display_cv(img, display_name=None):
    if not display_name: 
        display_name = str(random.randint(1, 100))
    temp_img = cv2.resize(img, (960, 450))
    cv2.imshow(display_name, temp_img)

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_image_blur(img):
    # image_blurred = cv2.GaussianBlur(image_y,(3,3),0)
    image_blurred = cv2.medianBlur(img, 31)
    # image_blurred = cv2.bilateralFilter(gray, 31, 61, 39)
    return image_blurred

def canny_edge_detection(img):
    return cv2.Canny(img, 1, 25, True)

def threshold_image(img):
    # cv2.waitKey(0)
    # tl = 100
    # ret, thresh = cv2.threshold(image_blurred, tl, 255, 0)
    # while util.white_percent(thresh) > 0.85:
    #     tl += 10
    #     ret, thresh = cv2.threshold(image_blurred, tl, 255, 0)

    ret, thresh = cv2.threshold(img, 100, 255, 0)
    return thresh

def get_contours(edges):
    return cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def filter_contours(img, contours):
    if not contours:
        return None

    return [cc for cc in contours if img_utils.contourOK(img, cc)]

def simplify_contours(contours):
    if not contours:
        return None

    simplified_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,0.001*cv2.arcLength(hull,True),True))

    return simplified_contours

def get_largest_contour(contours):
    if not contours:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    return contours[max_index]

def get_hough_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=250)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(doc1_lines, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return lines

def get_points_from_rectangle(rectangle):
    pts = cv2.boxPoints(rectangle)
    pts = pts.reshape(-1, 1, 2)
    pts = pts.astype(int)
    return pts
    
def is_valid_cheque_contour(rect):
    gold = 202 / 92

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    if W > H:
        ratio = W/H
    else:
        ratio = H/W
    
    error = abs(ratio - gold) / gold * 100
    # print(error)
    return error < 5

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, rectangle):
    pts = cv2.boxPoints(rectangle)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    max_width = max(int(width_a), int(width_b))
    # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # max_height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image, transform_matrix, (max_width, max_height))

    # return the warped image
    return warped

def autocrop(img, debug=False, send_steps=False):

    gray = convert_to_gray(img)
    image_blurred = get_image_blur(gray)

    edges = canny_edge_detection(image_blurred)
    contours, hierarchy = get_contours(edges)

    # thresh = threshold_image(gray)
    # contours, hierarchy = cv2.findContours(thresh, 1, 2)

    filtered_contours = filter_contours(img, contours)
    simplified_contours = simplify_contours(filtered_contours)
    largest_contour = get_largest_contour(simplified_contours)

    if largest_contour is None:
        if send_steps is True:
            image_final_contour = img.copy()
            cv2.putText(image_final_contour,"NO CONTOURS FOUND", (image_final_contour.shape[1]//3, image_final_contour.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 3,(255, 0, 0), 6)
            return img, image_blurred, edges, image_final_contour
        
    rectangle = cv2.minAreaRect(largest_contour)
    pts = get_points_from_rectangle(rectangle)

    if not is_valid_cheque_contour(rectangle):
        if send_steps is True:
            image_final_contour = img.copy()
            cv2.drawContours(image_final_contour, [pts], -1, (0, 0, 255), 5)
            cv2.putText(image_final_contour,"NO VALID CONTOUR FOUND", (image_final_contour.shape[1]//3, image_final_contour.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 3,(255, 0, 0), 6)
            return img, image_blurred, edges, image_final_contour
        return img

    # img_cropped = crop_minAreaRect(img, rectangle)
    img_cropped = four_point_transform(img, rectangle)

    if debug is True:
        # Debugging
        display_cv(image_blurred, display_name="Blurred")
        display_cv(edges, display_name="Canny Edge Detection")
        # display_cv(thresh, "Threshold")

        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
        display_cv(img_contours, display_name="Contours Raw")

        img_filtered_contours = img.copy()
        cv2.drawContours(img_filtered_contours, filtered_contours, -1, (0,255,0), 3)
        display_cv(img_filtered_contours, display_name="Contours Filtered")

        img_simplified_contours = img.copy()
        cv2.drawContours(img_simplified_contours, simplified_contours, -1, (0,255,0), 3)
        display_cv(img_simplified_contours, display_name="Simplified Contours")

        img_final = img.copy()
        cv2.drawContours(img_final, [pts], -1, (255, 0, 0), 2)
        display_cv(img_final, display_name="Final Contour")

        display_cv(img_cropped, display_name="Cropped Image")
        cv2.waitKey(0)

    if send_steps is True:
        image_final_contour = img.copy()
        cv2.drawContours(image_final_contour, [pts], -1, (0, 0, 255), 5)

        return img_cropped, image_blurred, edges, image_final_contour
    return img_cropped

if __name__ == '__main__':
    img = cv2.imread('./images/ab1_tilted.png')
    img_cropped = autocrop(img, debug=True)

    cv2.imshow("Uncropped", img)
    cv2.imshow("Cropped", img_cropped)
    cv2.waitKey(0)

"""
def crop_minAreaRect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    print(W, H, angle)

    if angle < -45:
        angle+=90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int((x2-x1)),int((y2-y1)))

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)    
    cv2.imshow("fsddfsdf", cropped)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
    return croppedRotated
"""