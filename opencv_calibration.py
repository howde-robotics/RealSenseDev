import numpy as np
import cv2 as cv
import glob

# img_to_read = cv.imread("/home/benjamin/dragoon/RealSenseDev/realsense_checker1_Color.jpg", flags=cv.IMREAD_UNCHANGED)
# cv.imshow("img", img_to_read)
# cv.waitKey(0)
# gray_img_to_read = cv.cvtColor(img_to_read, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", gray_img_to_read)
# cv.waitKey(0)
# ret, corners = cv.findChessboardCorners(gray_img_to_read, (8, 6), None)

# if ret:
#     print("hey")

# termination criteria
cbRow = 8
cbCol = 6
cbSideLengthInches = 5.91/6
cbSideLengthMM = cbSideLengthInches * 0.0254

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbRow*cbCol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbCol,0:cbRow].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/*.png')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (cbRow,cbCol), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Pattern Found")

        # scale object points by known side length
        objp *= cbSideLengthMM
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (cbRow,cbCol), corners2, ret)
        cv.imshow('img', img)
        # print(objp)
        cv.waitKey(0)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx, dist, rvecs, tvecs)

# save the calibration for later use
np.savez("calibration_output", intrinsics=mtx, distortion_coeffecients=dist, ext_rotation=rvecs, ext_translation=tvecs)
