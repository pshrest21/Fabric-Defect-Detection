import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


fabric1 = cv2.imread('0003.jpg',0)
fabric2 = cv2.imread('0012.jpg',0)
fabric3 = cv2.imread('0020.jpg',0)
fabric4 = cv2.imread('0041.jpg',0)
fabric5 = cv2.imread('0076.jpg',0)
fabric6 = cv2.imread('0106.jpg',0)
fabric7 = cv2.imread('0158.jpg',0)
fabric8 = cv2.imread('0192.jpg',0)

fabric5_new = cv2.imread('0076_new.jpg',0)


#ret, fabric5 = cv2.threshold(fabric5,100,255,cv2.THRESH_BINARY)



def dft(img):
    f = np.fft.fft2(img)  
    ft = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(ft))
    return ft, ms

def low_pass_filter(img,n):
    fshift, ms = dft(img)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    mask = np.zeros([rows, cols])
    mask[int(crow)-n:int(crow)+n, int(ccol)-n:int(ccol)+n] = 1
    
    fshift = fshift*mask    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back 

def high_pass_filter(img,n):
    
    fshift, ms = dft(img)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    fshift[int(crow)-n:int(crow)+n, int(ccol)-n:int(ccol)+n] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def displayFourierImage(img, fourier_transform, title):
    plt.subplot(121), plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),
    plt.imshow(fourier_transform, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()
    
    


dft1, ms = dft(fabric1)
dft1, ms2 = dft(fabric2)
dft1, ms3 = dft(fabric3)
dft1, ms4 = dft(fabric4)
dft1, ms5 = dft(fabric5)
dft1, ms5_new = dft(fabric5_new)


displayFourierImage(ms5, ms5_new, '5')




'''
#From research paper

#Perform the otsu thresholding
#ret, th = cv2.threshold(fabric1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


scaler = .5
img = cv2.imread("0003.jpeg")


imfft = np.fft.fft2(img)
mags = np.abs(np.fft.fftshift(imfft))
angles = np.angle(np.fft.fftshift(imfft))
visual = np.log(mags)
visual2 = (visual - visual.min()) / (visual.max() - visual.min())*255
cv2.imshow('Visual 2', visual2.astype(np.uint8))

cv2.imshow('fftimg4.jpg',visual2)

height,width,depth = img.shape
masking = np.zeros((height,width))

vis = visual2.astype(np.uint8)
edges = cv2.Canny(vis,50,180)
cv2.imshow('Gradient', edges)

lines = cv2.HoughLines(edges,1,np.pi/180,200,200,10)
for rho,theta in lines[0]:
   a = np.cos(theta)
   b = np.sin(theta)
   x0 = a*rho
   y0 = b*rho
   x1 = int(x0 + 1000*(-b))
   y1 = int(y0 + 1000*(a))
   x2 = int(x0 - 1000*(-b))
   y2 = int(y0 - 1000*(a))

   cv2.line(masking,(x1,y1),(x2,y2),(255,255,255),1)

   cv2.imshow('HoughLines', masking)
   
   
   
   '''

'''

imfft = np.fft.fft2(fabric1)
mags = np.abs(np.fft.fftshift(imfft))
angles = np.angle(np.fft.fftshift(imfft))
visual = np.log(mags)
visual2 = (visual - visual.min()) / (visual.max() - visual.min())*255
cv2.imshow('Visual 2', visual2.astype(np.uint8))
'''













 

    
    
def mean_filter(img, figure_size):
    new_image = cv2.blur(img, (figure_size, figure_size))
    return np.array(new_image)
    
def gaussian_filter(img, figure_size):
    new_image = cv2.blur(img, (figure_size, figure_size))
    return np.array(new_image) 

def canny_edge(img, figure_size):
    img = np.uint8(img)
    new_image = cv2.Canny(img, 100, 200)
    return np.array(new_image)

def erosion(img, kernel, n):
    #erosion using opencv inbuilt function cv2.erode()
    erosion = cv2.erode(img, kernel, iterations = n)
    return erosion
    
def dialation(img, kernel, n):
    dialation = cv2.dilate(img, kernel, iterations = n)
    return dialation


def histEqualize(img_array, cdf_list):
    final_dict = dict()
    cdf_list = np.rint(cdf_list)
    #print(cdf_list)
    for i in range(0, len(cdf_list)):
        final_dict[i] = int(cdf_list[i])
    #print(final_dict)    
    my_list = np.zeros([512, 512])
    #print(img_array)
    for i in range(0, len(img_array)):
        for j in range(0, len(img_array[i])):
            my_list[i][j] = final_dict[img_array[i][j]]
            
    #print(my_list)
    return my_list

def getImageInfo(img_array):
    #getting information from original image     
    dict_img = img_dict(img_array)
    my_list = sorted(dict_img.items())
    
    #set the keys to x and values to y
    x,y = zip(*my_list)   
    x = np.array(x)
    y = np.array(y) / 262144
    cdf_fish = cdf(x, y)
    
    return x,y,cdf_fish

def img_dict(img_array):  
    new_img_array = oneDList(img_array)
    #make a dictionary where the key is the grayscale and value is it's count
    my_dict = dict(Counter(new_img_array))
   
    #sort the dictionary in ascending order and convert it to list of tuples of key and value pairs
    my_list = sorted(my_dict.items())

    #set the keys to x and values to y
    x,y = zip(*my_list)   
    x = np.array(x)
    y = np.array(y)
  
    my_new_dict = dict()
    #have key and value pairs for all grayscales (0-255)
    for i in range(0, 256):
        if(i in x):
            my_new_dict[i] = my_dict[i]
        else:
            my_new_dict[i] = 0      
     
    return my_new_dict


def oneDList(img_array):    
    #create convert 2D array of images to 1D array
    new_img_array = [item for sublist in img_array for item in sublist] 
    return new_img_array

def cdf(my_list, y):
    cdf_list = []
    sum = y[0]
    cdf_list.append(sum)
    for i in range(1, len(y)):
        sum = sum + y[i]
        cdf_list.append(sum)
    
    cdf_list = np.array(cdf_list)
    return cdf_list

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,5))




#Using Morphological operations
#Fabric1-----------------------------

'''
#Erosion
erode = erosion(fabric5, kernel, 1)
erode1 = erosion(erode, kernel1, 1)
#cv2.imshow('eroded',erode)
#cv2.imshow('eroded1',erode1)

#Dilation
dilate = dialation(erode1, kernel2, 1)
cv2.imshow('input image',fabric5)

ret, img = cv2.threshold(dilate,120,255,cv2.THRESH_BINARY)
cv2.imshow('final', img)


#Fabric2----------------------------

#Erosion
erode = erosion(fabric2, kernel, 1)
erode1 = erosion(erode, kernel1, 1)
#cv2.imshow('eroded',erode)
cv2.imshow('original', fabric2)

dilate = dialation(erode1, kernel, 1)
cv2.imshow('final output',dilate)



#Fabric3---------------------------
#Erosion
erode = erosion(fabric3, kernel, 1)
erode1 = erosion(erode, kernel1, 1)
#cv2.imshow('eroded',erode)
cv2.imshow('original', fabric3)

dilate = dialation(erode1, kernel, 1)
cv2.imshow('final output',dilate)
'''


'''
#Histogram equalization part
x, y, cdf= getImageInfo(fabric1)
equalized_image1 = histEqualize(fabric1, cdf * 255) 
new_img = Image.fromarray(equalized_image1)
#new_img.convert('L').save('equalized.png', optimize = True)

'''


'''
n = 20
#apply the low pass filter to the images
img_back = low_pass_filter(fabric4, n)
#displayFourierImage(fabric4, img_back, 'Low Pass Filter')

img_back2 = high_pass_filter(fabric4, n)
#displayFourierImage(fabric4, img_back2, 'High Pass Filter')


mean_filtered = mean_filter(img_back, 5)
#displayFourierImage(img_back, mean_filtered, 'Mean Filtered')

gaussian_filtered = gaussian_filter(mean_filtered, 5)
#displayFourierImage(mean_filtered, gaussian_filtered, 'Gaussian Filtered')

#edge = canny_edge(img_back, 100)

ret, fabric = cv2.threshold(gaussian_filtered,195,255,cv2.THRESH_BINARY)

dilate = dialation(fabric, kernel, 2)
erode = erosion(dilate, kernel, 1)
'''






#cv2.imshow('final', erode)

cv2.waitKey(0)
cv2.destroyAllWindows()













