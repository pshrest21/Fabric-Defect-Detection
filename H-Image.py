import cv2
import numpy as np
from PIL import Image, ImageFilter 


def dft(img):
    f = np.fft.fft2(img)  
    ft = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(ft))
    return ft, ms

def high_pass_filter(img,n):
    
    fshift, ms = dft(img)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    fshift[int(crow)-n:int(crow)+n, int(ccol)-n:int(ccol)+n] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def HIPI(img):
    new_img = np.zeros((512, 512), dtype = np.uint8)
    for i in range(1, len(img)-1):
        for j in range(1, len(img[i])-1):
            sum = np.abs(img[i][j]-(img[i-1][j-1] + img[i-1][j] +
                    img[i-1][j+1] + img[i][j-1] + 
                    img[i][j] + img[i][j+1] + img[i+1][j-1]
                    + img[i+1][j] + img[i+1][j+1]))
            mean = sum / 9
            
            var = np.abs((img[i-1][j-1]-mean)**2 + (img[i-1][j]-mean)**2 +
                    (img[i-1][j+1]-mean)**2 + (img[i][j-1]-mean)**2 + 
                    (img[i][j]-mean)**2 + (img[i][j+1]-mean)**2 + (img[i+1][j-1]-mean)**2
                    + (img[i+1][j]-mean)**2 + (img[i+1][j+1]-mean)**2)
            
            var = var / 8
            new_img[i-1][j-1] = mean
        
    return new_img


def erosion(img, kernel, n):
    #erosion using opencv inbuilt function cv2.erode()
    erosion = cv2.erode(img, kernel, iterations = n)
    return erosion
    
def dialation(img, kernel, n):
    dialation = cv2.dilate(img, kernel, iterations = n)
    return dialation

def approach1(img, minFilter, threshold, binary_type, kernelD, n1, kernelE,n2, medianFilter, i):
    img = img.filter(ImageFilter.MinFilter (size = minFilter))
    img = np.array(img)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, new_img = cv2.threshold(new_img,threshold,255,binary_type) #convert to binary image
    new_img = dialation(new_img, kernelD, 1)
    new_img = erosion(new_img, kernelE, 1)
    new_img = cv2.medianBlur(new_img, medianFilter)
    #cv2.imshow('detect_'+str(i)+'.jpg', new_img)


def approach2(img, highPass, minFilter, threshold, binary_type, medianFilter, i):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = high_pass_filter(img, highPass)
    new_img = HIPI(img)
    new_img = np.array(new_img)
    new_img = Image.fromarray(new_img)
    new_img = new_img.filter(ImageFilter.MinFilter (size = minFilter))
    new_img = np.array(new_img)
    ret, new_img = cv2.threshold(new_img,threshold,255,binary_type)
    new_img  = cv2.medianBlur(new_img, medianFilter)
    cv2.imshow('detect_'+str(i)+'.jpg',new_img)

#--------------------Computation Part------------
#kernels to use
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


#Using approach 1
fabric1 = Image.open('1.jpg', 'r')
approach1(fabric1,3, 200, cv2.THRESH_BINARY,kernel,1,kernel,1,5, '1')

fabric2 = Image.open('2.jpg', 'r')
approach1(fabric2,3, 190, cv2.THRESH_BINARY,kernel1,1,kernel1,1,15, '2')

fabric4 = Image.open('4.jpg', 'r')
approach1(fabric4,3, 120, cv2.THRESH_BINARY_INV,kernel1,1,kernel1,1,7, '4')

fabric5 = Image.open('5.jpg', 'r')
approach1(fabric5,3, 120, cv2.THRESH_BINARY_INV,kernel1,1,kernel1,1,5, '5')

fabric6 = Image.open('6.jpg', 'r')
approach1(fabric6,3, 100, cv2.THRESH_BINARY,kernel1,1,kernel1,1,5, '6')

fabric8 = Image.open('8.jpg', 'r')
approach1(fabric8,3, 83, cv2.THRESH_BINARY_INV,kernel,1,kernel,1,5, '8')


#Using approach 2
fabric3 = Image.open('3.jpg', 'r')
approach2(fabric3, 50, 3, 21, cv2.THRESH_BINARY, 7, '3')


fabric7 = Image.open('7.jpg', 'r')
approach2(fabric7, 50, 3, 10, cv2.THRESH_BINARY, 11, '7')



cv2.waitKey(0)
cv2.destroyAllWindows()




















