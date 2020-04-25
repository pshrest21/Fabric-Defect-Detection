import cv2
import numpy as np

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

def gaussian_filter(img, figure_size):
    new_image = cv2.blur(img, (figure_size, figure_size))
    return np.array(new_image) 

def canny_edge(img, x,y, n):
    img = np.uint8(img)
    new_image = cv2.Canny(img, x, y, apertureSize=n)
    return np.array(new_image)

def erosion(img, kernel, n):
    #erosion using opencv inbuilt function cv2.erode()
    erosion = cv2.erode(img, kernel, iterations = n)
    return erosion
    
def dialation(img, kernel, n):
    dialation = cv2.dilate(img, kernel, iterations = n)
    return dialation



#--------------------Computation Part------------
fabric1 = cv2.imread('1.jpg',0)
fabric2 = cv2.imread('2.jpg',0)
fabric3 = cv2.imread('3.jpg',0)
fabric4 = cv2.imread('4.jpg',0)
fabric5 = cv2.imread('5.jpg',0)
fabric6 = cv2.imread('6.jpg',0)
fabric7 = cv2.imread('7.jpg',0)
fabric8 = cv2.imread('8.jpg',0)


#kernels to use
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,2))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,1))

'''
#----------------------Fabric1--------------------------
fabric1 = cv2.equalizeHist(fabric1) #equalize the image
fabric1 = high_pass_filter(fabric1, 50) #Use high pass filter to filter low frequency component
new_img = HIPI(fabric1) #get the H-Image
ret, new_img = cv2.threshold(new_img,20,255,cv2.THRESH_BINARY_INV) #convert to binary image

new_img = cv2.medianBlur(new_img, 5) #Removes the salt noise
cv2.imwrite('detect_1.jpg', new_img)


#----------------------Fabric2--------------------------
fabric2 = cv2.equalizeHist(fabric2) #equalize the image
fabric2 = high_pass_filter(fabric2, 50) #Use high pass filter to filter low frequency component
new_img = HIPI(fabric2) #get the H-Image
ret, new_img = cv2.threshold(new_img,20,255,cv2.THRESH_BINARY_INV) #convert to binary image

new_img = cv2.medianBlur(new_img, 3) #Removes the salt noise
cv2.imwrite('detect_2.jpg', new_img)

#----------------------Fabric3--------------------------
#Need to figure out different method


#----------------------Fabric4---------------------------
#fabric4 = cv2.equalizeHist(fabric4) #equalize the image
fabric4 = high_pass_filter(fabric4, 50) #Use high pass filter to filter low frequency component
new_img = HIPI(fabric4) #get the H-Image
ret, new_img = cv2.threshold(new_img,20,255,cv2.THRESH_BINARY) #convert to binary image

new_img = cv2.medianBlur(new_img, 5) #Removes the salt noise
new_img = erosion(new_img, kernel2, 1)
new_img = dialation(new_img, kernel3, 1)
cv2.imwrite('detect_4.jpg', new_img)

'''

#----------------------Fabric5---------------------------
fabric5 = cv2.equalizeHist(fabric5) #equalize the image
ret, new_img = cv2.threshold(fabric5,0,255,cv2.THRESH_BINARY_INV) #convert to binary image
new_img = cv2.medianBlur(new_img, 3) #Removes the salt noise
new_img = dialation(new_img, kernel3, 4) #dilates the image and joins the white points
cv2.imwrite('detect_5.jpg', new_img)






















'''
fabric1 = high_pass_filter(fabric1, 100)
new_img = HIPI(fabric1)
ret, new_img = cv2.threshold(new_img,10,255,cv2.THRESH_BINARY_INV)

new_img = erosion(new_img, kernel1, 3)
new_img = dialation(new_img, kernel1, 3)

cv2.imshow('defect_1.jpg', new_img)

for i in range(4, 9):
    fabric = cv2.imread(str(i)+'.jpg',0)
    #replicate = cv2.copyMakeBorder(fabric1,1,1,1,1,cv2.BORDER_REPLICATE)
    
    replicate = high_pass_filter(fabric, 50)
    
    new_img = HIPI(replicate)
    ret, new_img = cv2.threshold(new_img,10,255,cv2.THRESH_BINARY)
    
    new_img = erosion(new_img, kernel1, 3)
    new_img = erosion(new_img, kernel, 1)
    new_img = dialation(new_img, kernel2, 1) 
    new_img = dialation(new_img, kernel3, 1)
    
    cv2.imwrite('detect_'+str(i)+'.jpg',new_img)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()



























