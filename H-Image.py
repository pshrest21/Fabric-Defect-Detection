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





#--------------------Computation Part------------
#kernels to use
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,2))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,1))



#----------------------Fabric1--------------------------
fabric1 = cv2.imread('1.jpg', 0)
fabric1 = cv2.equalizeHist(fabric1) #equalize the image
ret, new_img = cv2.threshold(fabric1, 215,255,cv2.THRESH_BINARY) #convert to binary image
new_img = erosion(new_img, kernel2, 1) #removes excess noise from the image
new_img = cv2.medianBlur(new_img, 3) #Removes the salt noise
new_img = dialation(new_img, kernel3, 5) #dilates the image 5 times
cv2.imwrite('detect_1.jpg', new_img)





#----------------------Fabric2--------------------------
fabric2 = Image.open('2.jpg', 'r')
fabric2 = fabric2.filter(ImageFilter.MinFilter (size = 3))
fabric2 = np.array(fabric2)
new_img = cv2.cvtColor(fabric2, cv2.COLOR_BGR2GRAY)
ret, new_img = cv2.threshold(new_img,190,255,cv2.THRESH_BINARY) #convert to binary image
new_img = cv2.medianBlur(new_img, 11) #Removes the salt noise
new_img = erosion(new_img, kernel1, 1)
cv2.imwrite('detect_2.jpg', new_img)





#----------------------Fabric3--------------------------
fabric3 = cv2.imread('3.jpg',0)
ret, new_img = cv2.threshold(fabric3,150,255,cv2.THRESH_BINARY) #convert to binary image
new_img = cv2.medianBlur(new_img, 3) #Removes the salt noise
new_img = dialation(new_img, kernel1, 1) #Dilate the image
new_img = erosion(new_img, kernel1, 1) #Erode the image
new_img = cv2.medianBlur(new_img, 3) #Removes the salt noise
cv2.imwrite('detect_3_1.jpg', new_img)



#----------------------Fabric3 (Second Approach)--------------------------
fabric3 = cv2.imread('3.jpg',0)
fabric3 = high_pass_filter(fabric3, 50) #Use high pass filter to filter low frequency component
new_img = HIPI(fabric3) #get the H-Image
new_img = Image.fromarray(new_img)
new_img = new_img.filter(ImageFilter.MinFilter(size = 3)) #apply the min filter
new_img = np.array(new_img)
ret, new_img = cv2.threshold(new_img,21,255,cv2.THRESH_BINARY) #convert to binary image
new_img = cv2.medianBlur(new_img, 7) #Removes the salt noise
cv2.imwrite('detect_3_2.jpg', new_img)






#----------------------Fabric4---------------------------
fabric4 = Image.open('4.jpg', 'r')
fabric4 = fabric4.filter(ImageFilter.MinFilter (size = 3))
fabric4 = np.array(fabric4)
new_img = cv2.cvtColor(fabric4, cv2.COLOR_BGR2GRAY)
ret, new_img = cv2.threshold(new_img,120,255,cv2.THRESH_BINARY_INV) #convert to binary image
new_img = cv2.medianBlur(new_img, 7) #Removes the salt noise
cv2.imwrite('detect_4.jpg', new_img)







#----------------------Fabric5---------------------------
fabric5 = Image.open('5.jpg', 'r')
fabric5 = fabric5.filter(ImageFilter.MinFilter (size = 3))
fabric5 = np.array(fabric5)
new_img = cv2.cvtColor(fabric5, cv2.COLOR_BGR2GRAY)
ret, new_img = cv2.threshold(new_img,120,255,cv2.THRESH_BINARY_INV) #convert to binary image
new_img = cv2.medianBlur(new_img, 5) #Removes the salt noise
cv2.imwrite('detect_5.jpg', new_img)






#----------------------Fabric6---------------------------
fabric6 = Image.open('6.jpg', 'r')
fabric6 = fabric6.filter(ImageFilter.MinFilter(size = 3))
fabric6 = np.array(fabric6)
new_img = cv2.cvtColor(fabric6, cv2.COLOR_BGR2GRAY)
ret, new_img = cv2.threshold(new_img,100,255,cv2.THRESH_BINARY) #convert to binary image
new_img = cv2.medianBlur(new_img, 5) #Removes the salt noise
cv2.imwrite('detect_6.jpg', new_img)






#----------------------Fabric7---------------------------
fabric7 = cv2.imread('7.jpg',0)
fabric7 = high_pass_filter(fabric7, 50)
new_img = HIPI(fabric7)
new_img = Image.fromarray(new_img)
new_img = new_img.filter(ImageFilter.MinFilter (size = 3))
new_img = np.array(new_img)
ret, new_img = cv2.threshold(new_img,10,255,cv2.THRESH_BINARY)
new_img  = cv2.medianBlur(new_img, 11)
cv2.imwrite('detect_7.jpg',new_img)






#----------------------Fabric8---------------------------
fabric8 = Image.open('8.jpg', 'r')
fabric8 = fabric8.filter(ImageFilter.MinFilter (size = 3))
fabric8 = np.array(fabric8)
new_img = cv2.cvtColor(fabric8, cv2.COLOR_BGR2GRAY)
ret, new_img = cv2.threshold(new_img,85,255,cv2.THRESH_BINARY_INV) #convert to binary image
new_img = cv2.medianBlur(new_img, 5) #Removes the salt noise
cv2.imwrite('detect_8.jpg', new_img)



cv2.waitKey(0)
cv2.destroyAllWindows()





'''
new_ms = np.array(ms, dtype=np.uint8)
ret, new_img = cv2.threshold(new_ms,190,255,cv2.THRESH_BINARY) #convert to binary image
#new_img = cv2.medianBlur(new_img, 3) #Removes the salt noise

new_img = erosion(new_img, kernel5, 1)
new_img = dialation(new_img, kernel5, 5)

#cv2.imshow('dft', new_img)


lines = cv2.HoughLines(new_img,1,np.pi/180,200)
new_ms = cv2.UMat(new_ms)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(new_ms,(x1,y1),(x2,y2),(0,0,0),2)

cv2.imshow('Magnitude Spectrum after adding line', new_ms)

ms = ms.astype(np.uint8)
new_ms = new_ms.get()

new_ms = ms - new_ms

f = np.fft.fft2(new_ms)  
ft = np.fft.fftshift(f)
new_ms = 20 * np.log(np.abs(ft))

displayFourierImage(ms, new_ms, 'Defect area')

'''








































