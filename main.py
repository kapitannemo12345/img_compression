import matplotlib.pyplot as plt
import numpy as np
from operator import add
from functools import reduce
from tqdm import tqdm
import cv2
import scipy.fftpack
import os


def plot(img1, img2):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title('orginalny obraz')
    ax2 = fig.add_subplot(212)
    ax2.set_title('kompresja')
    # ax1.suptitle('orginał ', fontsize=16)
    ax1.imshow(img1)
    ax2.imshow(img2.astype(int))  # as type int sotosowac w zaleznosci od obrazka doc4 nie dziala z
    plt.show()

def plot2(img1,img2):
    Y = img1[:, :, 0]
    Cr = img1[:, :, 1]
    Cb = img1[:, :, 2]

    Y2 = img2[:, :, 0]
    Cr2 = img2[:, :, 1]
    Cb2 = img2[:, :, 2]

    fig, axs = plt.subplots(4, 2, sharey=True)
    fig.set_size_inches(9, 13)
    axs[0, 0].imshow(img1)
    axs[1, 0].imshow(Y, cmap=plt.cm.gray)
    axs[2, 0].imshow(Cr, cmap=plt.cm.gray)
    axs[3, 0].imshow(Cb, cmap=plt.cm.gray)

    axs[0, 1].imshow(img2)
    axs[1, 1].imshow(Y2, cmap=plt.cm.gray)
    axs[2, 1].imshow(Cr2, cmap=plt.cm.gray)
    axs[3, 1].imshow(Cb2, cmap=plt.cm.gray)

class container:
    def __init__(self,Chrom_type,qtype):
        self.y = 0
        self.x = 0
        self.chrom_type = Chrom_type
        self.quantization_type=qtype
        self.Y_list = []
        self.Cb_list = []
        self.Cr_list = []
        self.QY= np.array([
            [16, 11, 10, 16, 24,  40,  51,  61],
            [12, 12, 14, 19, 26,  58,  60,  55],
            [14, 13, 16, 24, 40,  57,  69,  56],
            [14, 17, 22, 29, 51,  87,  80,  62],
            [18, 22, 37, 56, 68,  109, 103, 77],
            [24, 36, 55, 64, 81,  104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ])
        self.QC = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ])
        self.ones = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])


def RGB_to_YCbCr(img):
    YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(int)
    return YCrCb


def YCbCr_to_RGB(img):
    RGB = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return RGB


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct(a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )


def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')


def chroma_subsampling(img):
    Y = img[:, :, 0]
    Cr = np.zeros((img.shape[0], int(img.shape[1] / 2)), dtype=img.dtype)
    Cb = np.zeros((img.shape[0], int(img.shape[1] / 2)), dtype=img.dtype)

    for y in range(0, img.shape[0]):
        c_x = 0
        for x in range(0, img.shape[1], 2):
            Cr[y, c_x] = img[y, x, 1]
            Cb[y, c_x] = img[y, x, 2]
            c_x = c_x + 1

    return Y, Cr, Cb


def chroma_resampling(Cr, Cb):
    print("rozmiar",Cr.shape[0],Cr.shape[1])

    h = Cr.shape[0]
    w = Cr.shape[1]*2
    t = Cr.dtype
    nCr = np.zeros((h, w), dtype=t)
    nCb = np.zeros((h, w), dtype=t)

    for y in range(0, h):
        c_x = 0
        for x in range(0, w, 2):
            nCr[y, x] = Cr[y, c_x]
            nCr[y, x + 1] = Cr[y, c_x]
            nCb[y, x] = Cb[y, c_x]
            nCb[y, x + 1] = Cb[y, c_x]
            c_x = c_x + 1

    return nCr, nCb


def quantization(block, con, switch):
    block2 = np.zeros([8, 8])

    if con.quantization_type == 1:
        if switch == 1:
            block2[:, :] = np.round(block[:, :] / con.QY).astype(int)
        if switch == 2:
            block2[:, :] = np.round(block[:, :] / con.QC).astype(int)

    if con.quantization_type == 2:
        block2[:, :] = np.round(block[:, :] / con.ones).astype(int)


    return block2

def de_quantization(block, con,switch):
    block2 = np.zeros([8, 8])
    if con.quantization_type == 1:
        if switch ==1:
            block[:, :] = block[:, :] * con.QY
        if switch == 2:
            block[:, :] = block[:, :] * con.QC

        #block[:, :, 2] = block[:, :, 2] * con.QC
    if con.quantization_type == 2:
        block[:, :] = block[:, :] * con.ones


    return block2

def zigzag(A):
    template = np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    #print("dlugosc a", len(A))
    if len(A.shape)==1:
        B = np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r, c] = A[template[r, c]]
    else:
        B = np.zeros((64,))
        for r in range(0, 8):
            for c in range(0, 8):
                B[template[r, c]] = A[r, c]
    return B

def RLE_encode(imgf):
    length = imgf.size
    ctr = 1
    RLE = np.array([])

    for i in range(0, length - 1):
        if imgf[i] == imgf[i + 1]:
            ctr = ctr + 1
            if i == length - 2:
                RLE = np.append(RLE, ctr)
                RLE = np.append(RLE, imgf[i])

        else:
            # a = np.array([imgf[i-1],ctr])
            # print(a)
            # RLE = np.append(RLE,a, axis=0)
            RLE = np.append(RLE, ctr)
            RLE = np.append(RLE, imgf[i])
            ctr = 1
        if i==length-2:
            if imgf[i] != imgf[i + 1]:
                RLE = np.append(RLE, ctr)
                RLE = np.append(RLE, imgf[i+1])


    return RLE


def RLE_decode(rle):
    #print("rle to",rle)
    out_array = []
    for i in range(0, len(rle), 2):
        out_array.extend([rle[i + 1] for j in range(int(rle[i]))])

    out_array = np.array(out_array)
    #print("po dekodowaniu",out_array)
    #print("dl poz dekodowaniu: ", len(out_array))
    return out_array


def encode3(img, con):
    block_img = np.zeros(img.shape)
    im_h, im_w = img.shape[:2]
    con.y = im_h
    con.x = im_w
    newImage = np.zeros([im_h, im_w, 3])
    bl_h, bl_w = 8, 8
    index = 0
    img2 = RGB_to_YCbCr(img)
    y = img2[:, :, 0]
    cb = img2[:, :, 1]
    cr = img2[:, :, 2]

    cb_y=im_h//2
    cb_x=im_w//2

    if con.chrom_type==1:
        cb_y=im_h//2
        cb_x=im_w//2
    if con.chrom_type==2:
        cb_y=im_h
        cb_x=im_w

    if con.chrom_type  == 1:
     y,cb,cr=chroma_subsampling(img2)

    newImage = np.zeros([im_h, im_w, 3])

    for row in tqdm(np.arange(im_h, step=bl_h)):
        for col in np.arange(im_w, step=bl_w):
            blocky = y[row:row + bl_h, col:col + bl_w]
            blocky = blocky - 128
            blocky = dct2(blocky)
            blocky = quantization(blocky, con, 1)
            ListY = zigzag(blocky)
            ListY = RLE_encode(ListY)
            con.Y_list.append(ListY)



    for row in tqdm(np.arange(im_h, step=bl_h)):
        for col in np.arange(cb_x, step=bl_w):
            blockcb = cb[row:row + bl_h, col:col + bl_w]
            blockcb = blockcb - 128
            blockcb = dct2(blockcb)
            blockcb = quantization(blockcb, con, con.quantization_type)
            ListCb = zigzag(blockcb)
            ListCb = RLE_encode(ListCb)
            con.Cb_list.append(ListCb)

            blockcr = cr[row:row + bl_h, col:col + bl_w]
            blockcr = blockcr - 128
            blockcr = dct2(blockcr)
            blockcr = quantization(blockcr, con, con.quantization_type)
            ListCr = zigzag(blockcr)
            ListCr = RLE_encode(ListCr)
            con.Cr_list.append(ListCr)




            # i1 = 0
            # i2 = 0
            # for i in range(row, row + bl_h):
            #
            #     i2 = 0
            #     for j in range(col, col + bl_w):
            #         newImage[i, j, :] = blocky[i1, i2, :]
            #         i2 = i2 + 1
            #     i1 = i1 + 1
            # index = index + 1
            #print("block2 ")
            #print(block2)

            #plot(img, newImage)







            index=index+1
    #plot(img, newImage)


def decode3(con, img):
    im_h = con.y
    im_w = con.x
    newImage = np.zeros([im_h, im_w , 3])

    #block = np.zeros([8, 8, 3])
    bl_h, bl_w = 8, 8
    index = 0
    #print("rozmiar list: ", len(con.Y_list))

    if con.chrom_type==1:
        cb_y=im_h
        cb_x=im_w//2
    if con.chrom_type==2:
        cb_y=im_h
        cb_x=im_w
    cb_t = np.zeros([cb_y, cb_x])
    cr_t = np.zeros([cb_y, cb_x])

    print(len(con.Y_list))
    print(len(con.Cr_list))
    print(len(con.Cb_list))

    for row in tqdm(np.arange(im_h, step=bl_h)):
        for col in np.arange(im_w, step=bl_w):
            #print("rozmiar indexu: ", index)

            ListY = con.Y_list[index]
            ListY = RLE_decode(ListY)
            blocky = zigzag(ListY)
            #block = np.dstack([ListY, ListCb, ListCr])

            de_quantization(blocky, con,1)

            blocky = idct2(blocky)
            blocky = blocky + 128

            i1=0
            i2=0
            for i in range(row, row+bl_h):
                i2=0
                for j in range(col, col+bl_w):
                    newImage[i, j, 0] = blocky[i1, i2]
                    i2 = i2+1
                i1 = i1 + 1

            index = index + 1
    index=0
    check=True
    for row in tqdm(np.arange(cb_y, step=bl_h)):
        for col in np.arange(cb_x, step=bl_w):

            ListCb = con.Cb_list[index]
            ListCr = con.Cr_list[index]

            ListCb = RLE_decode(ListCb)
            ListCr = RLE_decode(ListCr)

            blockcb = zigzag(ListCb)
            blockcr = zigzag(ListCr)

            de_quantization(blockcb, con, con.quantization_type)
            de_quantization(blockcr, con, con.quantization_type)

            blockcb = idct2(blockcb)
            blockcb = blockcb + 128

            blockcr = idct2(blockcr)
            blockcr = blockcr + 128

            if con.chrom_type == 2:
                i1 = 0
                i2 = 0
                for i in range(row, row + bl_h):
                    i2 = 0
                    for j in range(col, col + bl_w):
                        newImage[i, j, 1] = blockcb[i1, i2]
                        newImage[i, j, 2] = blockcr[i1, i2]
                        i2 = i2 + 1
                    i1 = i1 + 1
                index = index + 1
            if con.chrom_type == 1:
                i1 = 0
                i2 = 0
                for i in range(row, row + bl_h):
                    i2 = 0
                    for j in range(col, col + bl_w):
                        cb_t[i, j] = blockcb[i1, i2]
                        cr_t[i, j] = blockcr[i1, i2]
                        i2 = i2 + 1
                    i1 = i1 + 1
                index = index + 1

            #plot(img,newImage)
    if con.chrom_type == 1:
        #a=newImage[:, :, 2]
        #b=newImage[:, :, 1]
        ncr,ncb=chroma_resampling(cr_t, cb_t)
        newImage[:, :, 1] = ncb
        newImage[:, :, 2] = ncr

        #block = np.dstack([ListY, ListCb, ListCr])
    newImage2=YCbCr_to_RGB(newImage)
    return newImage2


con = container(2, 2)



img1 = plt.imread('b1.jpg')
#img = plt.imread('mk.jpg')
plt.imshow(img1)
#print(img2[0,0,2])
#print("yceb",img2)

encode3(img1, con)
img3=decode3(con,img1)
#plot(img1, img3)

#img4=img1[:512, :512, :]
#img5=img3[:512, :512, :]
#
img6=img1[512:768, :256, :]
img7=img3[512:768, :256, :]
#
img8=img1[1024:1280, :256, :]
img9=img3[1024:1280, :256, :]


#plt.imshow(img4)
#plt.imshow(img6)

#plot2(img1, img3)
#plot2(img4, img5)
plot2(img6, img7)
plot2(img8, img9)

plt.show()
# img2 = RGB_to_YCbCr(img1)
# chroma_sub(img2, 1)
# chroma_resam(img2, con.chrom_type)
# newImage2 = YCbCr_to_RGB(img2)
# plot(img1, newImage2 )


#plot(img1, img1)


#print(img2)
#print(img2.dtype)
#print(img2.shape)


#img3=YCbCr_to_RGB(img2)
#plot(img1, img2)
#plot(img1, img3)
#test_block=block_data(img1)
#plot(img1, test_block)










