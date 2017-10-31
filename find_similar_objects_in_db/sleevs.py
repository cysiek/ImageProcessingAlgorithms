import cv2
import numpy as np
from collections import OrderedDict
import operator


def process_image(img, face_pos, title):
	if len(face_pos) == 0:
		print 'No face found!'
		return
	mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) #create mask with the same size as image, but only one channel. Mask is initialized with zeros
	cv2.grabCut(img, mask, tuple(face_pos[0]), np.zeros((1,65), dtype=np.float64), np.zeros((1,65), dtype=np.float64), 1, cv2.GC_INIT_WITH_RECT) #use grabcut algorithm to find mask of face. See grabcut description for more details (it's quite complicated algorithm)
	mask = np.where((mask==1) + (mask==3), 255, 0).astype('uint8') #set all pixels == 1 or == 3 to 255, other pixels set to 0
	img_masked = cv2.bitwise_and(img, img, mask=mask) #create masked image - just to show the result of grabcut
	#show images
	cv2.imshow(title, mask) 
	cv2.imshow(title+' masked', img_masked)

	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert image to hsv
	channels = [0,1]
	channels_ranges = [180, 256]
	channels_values = [0, 180, 0, 256]
	histogram = cv2.calcHist([img_hsv], channels, mask, channels_ranges, channels_values) #calculate histogram of H and S channels
	histogram = cv2.normalize(histogram, None, 0, 255, cv2.NORM_MINMAX) #normalize histogram

	dst = cv2.calcBackProject([img_hsv], channels, histogram, channels_values, 1) # calculate back project (find all pixels with color similar to color of face)
	cv2.imshow(title + ' calcBackProject raw result', dst)

	ret, thresholded = cv2.threshold(dst, 25, 255, cv2.THRESH_BINARY) #threshold result of previous step (remove noise etc)
	cv2.imshow(title + ' thresholded', thresholded)
	
	cv2.waitKey(5000)
	#put partial results into one final image
	row1 = np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), img_masked))
	row2 = np.hstack((img_hsv, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)))
	return np.vstack((row1, row2))

def shirt_fft(img, face_pos, title):
	shirt_rect_pos = face_pos[0]
	# print shirt_rect_pos
	shirt_rect_pos[1] += 2*shirt_rect_pos[3] #move down (by 2 * its height) rectangle with face - now it will point shirt sample
	shirt_sample = img[shirt_rect_pos[1]:shirt_rect_pos[1]+shirt_rect_pos[3], shirt_rect_pos[0]:shirt_rect_pos[0]+shirt_rect_pos[2]].copy() #crop shirt sample from image
	shirt_sample = cv2.resize(shirt_sample, dsize=(256, 256)) #resize sample to (256,256)
	# cv2.imshow(title+' shirt sample', shirt_sample)

	shirt_sample_gray = cv2.cvtColor(shirt_sample, cv2.COLOR_BGR2GRAY) #convert to gray colorspace

	f = np.fft.fft2(shirt_sample_gray) #calculate fft
	fshift = np.fft.fftshift(f) #shift - now the brightest poitn will be in the middle
	# fshift = fshift.astype(np.float32)
	magnitude_spectrum = 20*np.log(np.abs(fshift)) # calculate magnitude spectrum (it's easier to show)
	print magnitude_spectrum.max(), magnitude_spectrum.min(), magnitude_spectrum.mean(), magnitude_spectrum.dtype
	magnitude_spectrum = cv2.normalize(magnitude_spectrum, alpha=255.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) #normalize the result and convert to 8uc1 (1 channe with 8 bits - unsigned char) datatype
	print magnitude_spectrum.max(), magnitude_spectrum.min(), magnitude_spectrum.mean(), magnitude_spectrum.dtype
	# cv2.imshow(title+' fft magnitude', magnitude_spectrum)
	magnitude_spectrum_original = magnitude_spectrum.copy()
	# temp, magnitude_spectrum = cv2.threshold(magnitude_spectrum, magnitude_spectrum.max()*0.75, 255.0, cv2.THRESH_TOZERO)
	# temp, magnitude_spectrum = cv2.threshold(magnitude_spectrum, 125, 255.0, cv2.THRESH_TOZERO)
	# temp, magnitude_spectrum = cv2.threshold(magnitude_spectrum, 250, 255.0, cv2.THRESH_TOZERO_INV) #clear the brightest part
	temp, magnitude_spectrum = cv2.threshold(magnitude_spectrum, 200, 255.0, cv2.THRESH_TOZERO) #clear all values from 0 to 200 - removes noise etc
	# cv2.imshow(title+' fft magnitude thresholded', magnitude_spectrum)
	# cv2.waitKey(1)

	# if chr(cv2.waitKey(5000)) == 'q':
		# quit()

	# return fshift
	return shirt_sample_gray, magnitude_spectrum_original, magnitude_spectrum

paths = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', 'plain1.jpg', 'plain2.jpg', 'plain3.jpg', 'plain4.jpg', 'stripes1.jpg', 'stripes2.jpg']
haar_cascade = cv2.CascadeClassifier('C:\\DevTools\\src\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml') #change it to path to face cascade - it's inside opencv folder

fft_dict = OrderedDict()
results_img = None

for path in paths:
	img = cv2.imread(path)
	face_pos = haar_cascade.detectMultiScale(img, 1.3, 5, cv2.CASCADE_FIND_BIGGEST_OBJECT)
	if len(face_pos) == 0: #if haar cascade failed to find any face, try again with different (more accurate, but slower) settings
		face_pos = haar_cascade.detectMultiScale(img, 1.1, 3, cv2.CASCADE_FIND_BIGGEST_OBJECT)
	# result = process_image(img, face_pos, path)
	# cv2.imwrite('result_' + path, result) #save the result
	results = shirt_fft(img, face_pos, path)
	if results_img is None:
		results_img = np.hstack(results)
	else:
		results_img = np.vstack((results_img, np.hstack(results)))
	fft_dict[path] = results[2]

similarity_dict = {}
cv2.imshow('results_img', results_img)
cv2.waitKey(1)


#for each image calcualte value of correlation with each other image
for i in range(len(fft_dict.keys())):
	for j in range(i+1, len(fft_dict.keys())):
	# for j in range(i, len(fft_dict.keys())):
		key1, key2 = fft_dict.keys()[i], fft_dict.keys()[j]
		print 'pair: ', key1, key2 
		img1 = fft_dict[key1]
		img2 = fft_dict[key2].copy()
		# img2 = img2[10:246, 10:246]
		correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
		# correlation = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)
		# print correlation
		print correlation.shape, correlation.dtype, correlation.max()
		similarity_dict[key1 + ' - ' + key2] = correlation.max()
		# similarity_dict[key1 + ' - ' + key2] = correlation

#sort values (from best to worst matches)
sorted_similarity_dict = sorted(similarity_dict.items(), key=operator.itemgetter(1), reverse=True)
print "final result: "
for a in sorted_similarity_dict:
	print a


cv2.waitKey(50000)