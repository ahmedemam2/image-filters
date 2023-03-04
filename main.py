import numpy as np
import cv2
from tkinter import *
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()


def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            (array[i], array[j]) = (array[j], array[i])
    (array[i + 1], array[high]) = (array[high], array[i + 1])
    return i + 1


def quicksort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quicksort(array, low, pi - 1)
        quicksort(array, pi + 1, high)
        return array

imageinput = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
image = imageinput
avg=0
maximum = -10000
minimum = 10000
ct=0
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        avg += image[y][x]
        ct+=1
avg = avg/ct

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        if maximum < image[y][x]:
            maximum = image[y][x]
        if minimum > image[y][x]:
            minimum = image[y][x]

img = cv2.imread('VeryExpensive.png',0)
top = Tk()
top.title("Image Processing filters")
top.geometry("500x150")



def calcavg(image):

    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    k = 3
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            sub_image = image_padded[y: y+k, x: x+k]
            img = sub_image.flatten()
            f = img.sum()
            d = f/len(img)
            output[y , x] = d
    cv2.imshow('Average Filter',output)


def indexpixels(arr, image, i, j, rangenum):
    for w in range(rangenum):
        arr.append(image[i][j + w])


def alphatrimmedmean(image):
    img = image.copy()
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            arr = []
            for w in range(3):
                indexpixels(arr, image, i - 1 + w, j - 1, 3)

            arrlen = len(arr) - 1
            arr = quicksort(arr, 0, arrlen)
            midIndex = int(arrlen / 2)
            total = addmidInd(arr, midIndex)
            img[i][j] = total

    cv2.imshow('aa', img)
    cv2.waitKey(0)


def addmidInd(arr, midIndex):
    total = 0
    for i in range(5):
        total += arr[midIndex + i - 2]
    total = int(total / 5)
    return total

def contraharmonic(image):
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    q = -1
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # output[y, x]=(kernel * image_padded[y: y+3, x: x+3]).sum()
            img_test = image_padded[y:y + 3, x: x + 3].copy()
            img = img_test.flatten()
            img2 = img_test.flatten()
            q1 = q + 1

            for i in range(0, len(img)):
                img[i] = img[i] ** q1
            for i in range(0, len(img2)):
                img2[i] = img2[i] ** q

            sum_1 = np.sum(img)
            sum_2 = np.sum(img2)
            if (sum_2 != 0):
                d = sum_1 / sum_2
                output[y, x] = d
    cv2.imshow("contraharmonic",output)

def AdaptiveMed(image):
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            flag = 0
            k = 3
            while (flag != 1 and x + k < image.shape[1] and y + k < image.shape[0]):
                sub_image = image_padded[y: y + k, x: x + k]
                img = sub_image.flatten()
                r_max = img.max()
                r_min = img.min()
                median_pos = int(len(img) / 2)
                img = np.sort(img)
                r_med = img[median_pos]
                mid = int(k / 2)
                r_xy = sub_image[mid, mid]
                if (r_med > r_min and r_med < r_max):
                    if (r_xy > r_min and r_xy < r_max):
                        sub_image[mid, mid] = r_xy
                        flag = 1
                    else:
                        flag = 1
                        sub_image[mid, mid] = r_med
                elif (flag == 0):
                    if (k < 9):
                        k += 2
                    else:
                        sub_image[mid, mid] = r_xy
                        flag = 1
    cv2.imshow('Adpative Median',image_padded)
def calcmedian(image):
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    k = 3
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            sub_image = image_padded[y: y + k, x: x + k]
            img = sub_image.flatten()
            img = quicksort(img,0,len(img)-1)
            d = img[4]
            output[y, x] = d
    cv2.imshow('Median Filter',output)


def HistogramEqualization(img):
    map_num = {}
    total_pixels = img.shape[0] * img.shape[1]
    map_pdf = {}
    map_sk = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] in map_num:
                map_num[img[i][j]] += 1
            else:
                map_num[img[i][j]] = 0

    for i in range(256):
        if (i not in map_num):
            map_num[i] = 0

    map_num_sorted = OrderedDict(sorted(map_num.items()))
    for f in map_num_sorted:
        map_pdf[f] = map_num_sorted[f] / total_pixels

    total = 0
    for i in map_pdf:
        total += (map_pdf[i] * 255)
        map_sk[i] = total

    for i in map_sk:
        map_sk[i] = int(round(map_sk[i], 0))

    img2 = img.copy()
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i][j] = map_sk[img2[i][j]]

    cv2.imshow('1',img)
    cv2.imshow('2',img2)
    x = []
    y = []
    for f in map_pdf:
        x.append(f)
        y.append(map_pdf[f])

    plt.plot(x, y)

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Before Equalization')
    plt.show()

    x = []
    y = []
    # x axis values
    for f in map_sk:
        x.append(f)
        y.append(map_pdf[map_sk[f]])

    plt.plot(x, y)

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('After Equalization')
    plt.show()



def gaussianNoise(image):
    gaussian = np.random.normal(loc=0, scale=1, size=image.size)
    gaussian = gaussian.reshape(image.shape[0], image.shape[1]).astype('uint8')
    noisy_image = cv2.add(image, gaussian)
    cv2.imshow('GaussN' , noisy_image)


def CalcAvg2(img):
    total_pixels = img.shape[0] * img.shape[1]
    total = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            total += img[i][j]

    avg = total / total_pixels
    return avg


def CalcDev2(img, avg):
    total_pixels = img.shape[0] * img.shape[1]
    total = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            diff = img[i][j] - avg
            total += math.pow(diff, 2)

    variance = total / total_pixels
    deviation = math.sqrt(variance)
    return deviation


def CalculateAvg(img):
    map_num = [0] * 256
    total_pixels = img.shape[0] * img.shape[1]
    map_pdf = [0] * 256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            map_num[int(img[i][j])] += 1

    for f in range(len(map_num)):
        map_pdf[f] = map_num[f] / total_pixels
    total = 0
    for h in range(len(map_pdf)):
        total += (map_pdf[h] * h)
    Li = []
    Li.append(total)
    Li.append(map_pdf)
    return Li


def CalculateDeviation(avg, pdf):
    variance_tot = 0
    for f in range(len(pdf)):
        diff = f - avg
        variance_tot += (math.pow(diff, 2) * pdf[f])
    deviation = math.sqrt(variance_tot)
    return deviation


def CheckCondition(k0, k1, k2, global_avg, local_avg, global_dev, local_dev):
    if (local_avg <= k0 * global_avg and (local_dev >= k1 * global_avg and local_dev <= k2 * global_dev)):
        return True
    else:
        return False


def working(image, k0, k1, k2, E):
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image.copy()
    Global_LI = CalculateAvg(image)
    global_avg = Global_LI[0]
    global_pdf = Global_LI[1]
    # global_avg = CalcAvg2(image)
    global_deviation = CalculateDeviation(global_avg, global_pdf)
    # global_deviation = CalcDev2(image , global_avg)
    check = False
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            img_test = image_padded[y: y + 3, x: x + 3].copy()
            LF = CalculateAvg(img_test)
            local_avg = LF[0]
            local_deviation = CalculateDeviation(LF[0], LF[1])
            check_case = CheckCondition(k0, k1, k2, global_avg, local_avg, global_deviation, local_deviation)
            if (check_case == True):
                image[y][x] = E * image[y][x]
    cv2.imshow('Histogram Statistics',image)



def adaptive_median_filter(img):
    filtered_image = np.zeros_like(img)
    padded_image = np.pad(img, 1, mode='symmetric')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window_size = 3
            while True:
                window = padded_image[i:i+window_size*2+1, j:j+window_size*2+1]
                median = np.median(window)
                if img[i, j] > np.min(window) and img[i, j] < np.max(window):
                    filtered_image[i, j] = img[i, j]
                    break
                elif window_size >= padded_image.shape[0] or window_size >= padded_image.shape[1]:
                    filtered_image[i, j] = median
                    break
                else:
                    window_size += 2
    cv2.imshow('Adaptive Median',filtered_image)

b1 = Button(top,text="avg",command=lambda:calcavg(imageinput)).grid(row=0,column=0,padx=10,pady=10)

b2 = Button(top,text="median",command=lambda:calcmedian(imageinput)).grid(row=0,column=1,padx=10,pady=10)

b3 = Button(top,text="Adaptive_median",command=lambda:adaptive_median_filter(imageinput)).grid(row=0,column=2,padx=10,pady=10)

b4 = Button(top,text="contra_harmonic",command=lambda:contraharmonic(imageinput)).grid(row=0,column=3,padx=10,pady=10)

b5 = Button(top,text="gausN",command=lambda:gaussianNoise(imageinput)).grid(row=1,column=0,padx=10,pady=10)

b6 = Button(top,text="HistEqualization",command=lambda:HistogramEqualization(imageinput)).grid(row=1,column=1,padx=10,pady=10)

b7 = Button(top,text="HistStats",command=lambda:working(imageinput,0.4,0.02,0.4,9)).grid(row=1,column=2,padx=10,pady=10)

b8 = Button(top,text="alphatrimmedmean",command=lambda:alphatrimmedmean(imageinput)).grid(row=1,column=3,padx=10,pady=10)



top.mainloop()

