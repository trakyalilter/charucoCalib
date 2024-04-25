import argparse
import numpy as np
import os
import imutils
import time

class Calibration():
    def __init__(self,cv2,path):
        self.cv2 = cv2
        self.path = path
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
        self.board = cv2.aruco.CharucoBoard((6,10),0.02,0.01,self.dictionary)
        self.params = cv2.aruco.DetectorParameters()
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        self.boardImage = self.board.generateImage((900,1600),10,1)
    
    def saveBoardImage(self):
        self.cv2.imwrite("BoardImage.jpg",self.boardImage)

    def detectCharucoBoardWithoutCalibration(self):
        inputVideo = self.cv2.VideoCapture(0)
        self.params.cornerRefinementMethod = self.cv2.aruco.CORNER_REFINE_NONE
        while inputVideo.grab():
            ret, frame = inputVideo.read()
            (markerCorners,markerIds,rejected) = self.cv2.aruco.detectMarkers(frame,self.dictionary,parameters=self.params)
            if len(markerCorners)>0:
                frame = self.cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                _,charucoCorners,charucoIds = self.cv2.aruco.interpolateCornersCharuco(markerCorners,markerIds,frame,self.board)
                if(charucoIds is not None):
                    if len(charucoIds)>0:
                        self.cv2.aruco.drawDetectedCornersCharuco(frame,charucoCorners,charucoIds,(255,0,0))
                markerIds = markerIds.flatten()
            frame = imutils.resize(frame, width=1200)
            self.cv2.imshow('charuco',frame)
            if self.cv2.waitKey(1) & 0xFF == ord('q'):
                break
        inputVideo.release() 
        self.cv2.destroyAllWindows() 

    def calibrateCamera(self):
        PATH_TO_YOUR_IMAGES = self.path
        image_files = []
        print(image_files)
        
        image_files.sort()

        all_charuco_corners = []
        all_charuco_ids = []
        
        for image_file in image_files:
            img = self.cv2.imread(image_file)
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            #edges = cv2.Canny(gray,240,300)
            #cv2.imshow("preimage",gray)
            #cv2.waitKey(250)
            #cv2.imshow("canny",edges)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = self.cv2.filter2D(img, -1, kernel)
            #cv2.imshow("sharpened",sharpened)
            #cv2.waitKey(500)
            blurred = self.cv2.GaussianBlur(gray, (3, 3), 0)
            self.cv2.imshow("blurred",blurred)
            edges = self.cv2.Canny(blurred,200,300)
            gray = blurred
            #blurred = cv2.GaussianBlur(img, (15, 15), 5)
            image_copy = gray.copy()
            marker_corners, marker_ids, _ = self.cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
            
        # Iterate through calibration images
            if len(marker_corners) > 0:
                    image_copy = self.cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                    charuco_retval, charuco_corners, charuco_ids = self.cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, self.board)
                    if (charuco_retval):
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
        if len(image_files>0):
            retval, camera_matrix, dist_coeffs, rvecs, tvecs = self.cv2.aruco.calibrateCameraCharuco(all_charuco_corners,
                                                                                                all_charuco_ids,
                                                                                                self.board,
                                                                                                img.shape[:2],
                                                                                                None, None,
                                                                                                criteria=self.criteria)
            np.save('char_camera_matrix.npy', camera_matrix)
            np.save('char_dist_coeffs.npy', dist_coeffs)
            return retval, camera_matrix, dist_coeffs
