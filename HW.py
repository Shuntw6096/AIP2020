from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from tkinter.simpledialog import askstring, askfloat, askinteger
from sys import exit
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt



class gui:
  '''create tkinter gui'''
  def __init__(self):
    '''
    self.window: tkinter root window
    self.labels: list, input label and output label
    '''
    self.window = tk.Tk()
    self.window.title('AIP M10902116')
    self.labels = []
    LoadImage(self.window, self.labels)
    ImageHistogram(self.window, self.labels)
    GaussianWhiteNoise(self.window, self.labels)
    WaveletTransform(self.window, self.labels)
    HistogramEqualization(self.window, self.labels)
    SmoothingConvolution(self.window, self.labels)
    self.window.protocol("WM_DELETE_WINDOW", self.destroy)
    self.window.mainloop()

  def destroy(self):
    '''destroy process'''
    for l in self.labels:
      l.destroy()
    self.labels.clear()
    self.window.destroy()
    exit()

  @staticmethod
  def resize_label_image(labels, window):
    '''
    resize image size in window to make UI fixed.
    labels: list, fix label's image.
    '''
    img_x_location=0
    for idx, l in enumerate(labels):
      if hasattr(l, 'rphoto'):
        img = l.rphoto.image
        w, h = img.size
        ratio = w / 500
        # Rounding
        h = int(h//ratio) + 1 if (h/ratio - h//ratio) >= 0.5 else int(h//ratio)
        if h > 750:
          # limit height
          h = 750
        if ratio > 1:
          w = 500
          # downsampling
          img = img.resize((w, h), Image.ANTIALIAS)

        # not to do upsampling
        photo = ImageTk.PhotoImage(image=img)
        photo.image = img
        l.configure(image=photo)
        l.photo = photo
        if idx == 0:
          # input image
          l.place(x=0, y=20, relx=.1, rely=.1)
          img_x_location += img.width
        elif idx == 1:
          # output image
          l.place(x=img_x_location, y=20, relx=.12, rely=.1)

    # keep window
    window.mainloop()


class LoadImage:
  '''load image to show'''
  def __init__(self, window, labels):
    self.window = window
    self.labels = labels
    load_btn = tk.Button(self.window, text="Load Image")
    load_btn.place(x=5, y=5)
    load_btn.bind('<Button-1>', self.load_image)


  def load_image(self, event):
    '''press load image buttom to ask file and show it on window.'''

    # clear window -- delete image on window
    for l in self.labels:
      l.destroy()
    self.labels.clear()

    # get file path(string)
    image = askopenfilename(
        title="Select file",
        filetypes=(
        ("jpeg files", "*.jpg"),
        ("bmp file", "*.bmp"),
        ("ppm file", "*.ppm"),
        ("png file", "*.png",)
        )
      )
    # open image from path
    image = Image.open(image)
    # convert image format type
    if image.mode == "RGBA":
      image = image.convert("RGB")

    # show input image on window two times, and add labels to list which can be accessed by other function.
    self.canvas = tk.Canvas(self.window)
    photo = ImageTk.PhotoImage(image=image)
    photo.image = image

    input_label = tk.Label(image=photo)
    input_label.rphoto = photo
    self.labels.append(input_label)

    output_label = tk.Label(image=photo)
    output_label.rphoto = photo
    self.labels.append(output_label)
    # image layout
    gui.resize_label_image(self.labels, self.window)


class ImageHistogram:
  '''show input image's grayscale histogram.'''
  def __init__(self, window, labels):
    self.window = window
    self.labels = labels
    func_btn = tk.Button(self.window, text="Histogram")
    func_btn.place(x=120, y=5)
    func_btn.bind('<Button-1>', self.histogram)

  def histogram(self, event):
    '''
    from label get input image(RGB, normally) and convert it to grayscale,  
    update label's image to show  grayscale image, and show histogram.  
    self.labels[0].rphoto.image: input image  
    self.labels[0].rphoto.image: output image  
    '''
    # convert RGB to grayscale
    image = self.labels[0].rphoto.image.convert('L')
    # reshape image to 1-d array to compute histogram
    hist = np.array(image).reshape(-1)
    # build histogram
    fig = plt.figure()
    plt.hist(hist, np.arange(257, dtype=int), facecolor='g', alpha=0.75)
    plt.xlabel('GrayScale')
    plt.ylabel('Quantity')
    plt.title('Histogram of GrayScale')
    plt.grid(True)
    # convert histogram(image) to array, and array to PIL.Image
    img_hist = self.array2image(self.figure2array(fig))

    # update label images on window
    pto_img_gr = ImageTk.PhotoImage(image=image)
    pto_img_gr.image = image
    self.labels[0].configure(image=pto_img_gr)
    self.labels[0].rphoto = pto_img_gr

    pto_img_hist = ImageTk.PhotoImage(image=img_hist)
    pto_img_hist.image = img_hist
    self.labels[1].configure(image=pto_img_hist)
    self.labels[1].rphoto = pto_img_hist
    # image layout
    gui.resize_label_image(self.labels, self.window)

  @staticmethod
  def figure2array(figure):
    '''
    convert plt.figure to ndarray(RGB).  
    input:
      figure: plt.figure  
    return:
      ndarray, shape=(w,h,3), dtype=np.uint8  
    '''
    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf

  @staticmethod
  def array2image(arr):
    '''
    Convert arr to bytes, and open image from bytes, if using Image.fromarray directly, image will distort.  
    input:
      arr: ndarray(w, h), dtype=np.uint8  
    return:
      PIL.Image  
    '''
    w, h, _ = arr.shape
    return Image.frombytes("RGB", (w, h) , arr.tobytes())
    

class GaussianWhiteNoise:
  ''' Add Gaussian White Noise to image'''
  def __init__(self, window, labels):
    self.window = window
    self.labels = labels
    func_btn = tk.Button(self.window, text="Gaussian White Noise")
    func_btn.place(x=230, y=5)
    func_btn.bind('<Button-1>', self.add_noise)

  def add_noise(self, event):
    deviation = askfloat("Set up noise factor", "Noise Deviation: ([0, 100])", minvalue=0., maxvalue=100., initialvalue=50.)
    input_array = np.array(self.labels[0].rphoto.image)

    if len(input_array.shape) == 3:
      # RGB image
      h, w, _ = input_array.shape
      noise_matrix = deviation * self.get_noise_array(h ,w)
    else:
      # grayscale image
      h, w = input_array.shape
      noise_matrix = deviation * self.get_noise_array(h ,w, isRGB=False)

    # build gaussian noise histogram
    fig = plt.figure()
    plt.hist(noise_matrix.reshape(-1), 50, facecolor='g', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Quantity')
    plt.title('Histogram of Gaussian White Noise')
    plt.grid(True)

    input_array = input_array.astype(np.float64)
    input_array += noise_matrix
    # restrict to [0, 255]
    input_array[input_array > 255] = 255
    input_array[input_array < 0] = 0
    output = input_array.astype(np.uint8)
    output = Image.fromarray(output)

    # convert histogram(image) to array, and array to PIL.Image
    img_hist = ImageHistogram.array2image(ImageHistogram.figure2array(fig))

    # update label images on window
    pto_output = ImageTk.PhotoImage(image=output)
    pto_output.image = output
    self.labels[0].configure(image=pto_output)
    self.labels[0].rphoto = pto_output

    pto_img_hist = ImageTk.PhotoImage(image=img_hist)
    pto_img_hist.image = img_hist
    self.labels[1].configure(image=pto_img_hist)
    self.labels[1].rphoto = pto_img_hist

    # image layout
    gui.resize_label_image(self.labels, self.window)


  def get_noise_array(self, h, w, isRGB=True):
    '''
    produce noise matrix and return it.
    '''
    if w % 2 == 0:
      w = w // 2
      need_padding = False
    else:
      w = (w - 1) // 2
      need_padding = True

    if isRGB:
      # RGB
      phi = np.random.uniform(size=(h, w, 3))
      r = np.random.uniform(size=(h, w, 3))
    else:
      # grayscale
      phi = np.random.uniform(size=(h, w))
      r = np.random.uniform(size=(h, w))

    # duplicate columns
    phi = np.repeat(phi, 2, axis=1)
    r = np.repeat(r, 2, axis=1)
    # even colunm index make cos
    phi[:, ::2] = np.cos(-2 * np.pi * phi[:, ::2])
    # odd colunm index make sin
    phi[:, 1::2] = np.sin(-2 * np.pi * phi[:, 1::2])
    r = (-2 * np.log(r)) ** 0.5
    noise_mat = phi * r

    if need_padding:
      # padding width
      if isRGB:
        # RGB
        noise_mat = np.pad(noise_mat, ((0, 0), (0, 1), (0, 0)), 'constant')
      else:
        # grayscale
        noise_mat = np.pad(noise_mat, ((0, 0), (0, 1)), 'constant')

    return noise_mat


class WaveletTransform:
  '''Haar Wavelet Transform'''

  def __init__(self, window, labels):
    self.window = window
    self.labels = labels
    func_btn = tk.Button(self.window, text="Haar Wavelet Transform")
    func_btn.place(x=410, y=5)
    func_btn.bind('<Button-1>', self.apply_transform)

  def apply_transform(self, event):
    image = self.labels[0].rphoto.image
    # Determine if it is a grayscale image. If it is not, turn to grayscale.
    if not image.mode == 'L':
      image = image.convert('L')

    is_power_of_2 = lambda x: np.log2(x) - int(np.log2(x)) == 0

    # Determine if the size of the image is a square and the one side is power of 2. 
    # If it is a square, use the nearest power of 2 as the size; 
    # if not, use the power of 2 closest to the short side as the size.
    w, h = image.size
    if w == h:
      if not is_power_of_2(w):
        size, is_downsampling = self.determine_size(w)
        if is_downsampling:
          image = image.resize((size, size), Image.ANTIALIAS)
        else:
          image = image.resize((size, size))
    else:
      size = w if w < h else h
      if not is_power_of_2(size):
        size, is_downsampling = self.determine_size(size)
        if is_downsampling:
          image = image.resize((size, size), Image.ANTIALIAS)
      image = image.resize((size, size))
    maxtimes = int(np.log2(image.size[0]))
    times = tk.simpledialog.askinteger("Set up Wavelet Transformation times", f"Times: ([1, {maxtimes}])", minvalue=1, maxvalue=maxtimes, initialvalue=2)
    input_array = np.array(image)
    output_array = self.transform(input_array.astype(np.float64), times)

    output_image = Image.fromarray(output_array.astype(np.uint8))
    pto_input = ImageTk.PhotoImage(image=image)
    pto_input.image = image
    self.labels[0].configure(image=pto_input)
    self.labels[0].rphoto = pto_input

    pto_output = ImageTk.PhotoImage(image=output_image)
    pto_output.image = output_image
    self.labels[1].configure(image=pto_output)
    self.labels[1].rphoto = pto_output

    # image layout
    gui.resize_label_image(self.labels, self.window)

  def transform(self, arr, times):
    '''
    Do transformation for many times, and paste result in place.  
    input:
      arr: 2-d array.
      times: int, do transformation for many times.
    output:
      output: 2-d array
    '''
    if times != 1:
      size = arr.shape[0]
      output = np.zeros_like(arr, dtype=arr.dtype)
      for _ in range(times):
        result = self.transform(arr, 1)
        output[:size, :size] = result
        size //= 2
        arr = output[:size, :size]
      return output

    size = arr.shape[0]
    output = np.zeros_like(arr, dtype=arr.dtype)

    for mode in ["LL", "HL","HH", "LH"]:
      result = self.filter2D(mode, arr)
      # build border at lower left corner
      result[:, 0] = 255
      result[-1, :] = 255
      if mode == "LL":
        output[:size//2, :size//2] = result
      elif mode == "HL":
        result *= 2.5
        output[:size//2, size//2:] = result
      elif mode == "HH":
        result *= 3
        output[size//2:, size//2:] = result
      elif mode == "LH":
        result *= 2.5
        output[size//2:, :size//2] = result

    output[output > 255] = 255
    return output

  def filter2D(self, mode, arr):
    '''
    Filter first in the x direction, then filter in the y direction,  
    only make two dimensional filtering on a block.  
    input:  
      mode: str, only four choices, "LL", "LH", "HH", "HL"  
      arr: 2-d array  
    output:  
      2-d array
    '''
    if mode == "LL":
      filters = [np.sum, np.sum]
    elif mode == "LH":
      filters = [np.sum, np.diff]
    elif mode == "HL":
      filters = [np.diff, np.sum]
    elif mode == "HH":
      filters = [np.diff, np.diff]

    arr = arr.copy()
    for idx, f in enumerate(filters):
      if idx == 0:
        # x direction, equivalent to
        # kernel size(h,w): (1, 2)
        # stride(h,w): (1, 2) 
        strides = (arr.strides[0], arr.strides[1] * 2, arr.strides[1])
        shape = (arr.shape[0], arr.shape[1] // 2, 2)
      else:
        # y direction, equivalent to
        # kernel size(h,w): (2, 1)
        # stride(h,w): (2, 1) 
        strides = (arr.strides[1], arr.strides[0] * 2, arr.strides[0])
        shape = (arr.shape[1], arr.shape[0] // 2, 2)
      arr3D = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
      if f == np.diff:
        # Forward difference
        arr = -0.5 * f(arr3D, axis=2)[..., 0] # 2D array
      else:
        arr = 0.5 * f(arr3D, axis=2) # 2D array

    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr.T

  def determine_size(self, length):
    '''
    The new image size is determined by a power of 2 closest to the original image size.  
    input:
      length: int, this integer is a power of 2 impossible.
    output:
      size: int, a power of 2 closest to the original image size.
      is_downsampling: bool, downsampling to new size.
    '''
    size = 2
    while True:
      new_size = size * 2
      if new_size < length:
        size = new_size
      elif abs(new_size - length) >= abs(size - length):
        return size, True
      elif abs(new_size - length) < abs(size - length):
        return new_size, False


class HistogramEqualization:
  ''' Histogram Equalization '''
  def __init__(self, window, labels):
    self.window = window
    self.labels = labels
    func_btn = tk.Button(self.window, text="Histogram Equalization")
    func_btn.place(x=610, y=5)
    func_btn.bind('<Button-1>', self.equalization)
  
  def equalization(self, event):
    image = self.labels[0].rphoto.image
    if not image.mode == 'L':
      image = image.convert('L')
    w, h = image.size
    hist = np.array(image).reshape(-1)
    grayscale_level = 256

    # build original histogram
    original_fig = plt.figure()
    pixels_each_grayscales, _, _ = plt.hist(hist, np.arange(grayscale_level + 1, dtype=int), facecolor='g', alpha=0.75)
    plt.xlabel('GrayScale')
    plt.ylabel('Quantity')
    plt.title('Histogram of GrayScale')
    plt.grid(True)

    # reszie original histogram image and cancatenate images vertically
    original_hist_img = ImageHistogram.array2image(ImageHistogram.figure2array(original_fig))
    original_hist_img = self.resize(image, original_hist_img)

    # build cumulative histogram
    cumulative_hist = pixels_each_grayscales.astype(int).cumsum()
    # g_min is minimum grayscale for which cumulative_hist's pixel number > 0
    g_min = np.argwhere(cumulative_hist>0).min()
    grayscale_transform_arr = np.zeros((grayscale_level),dtype=int)
    for i in range(grayscale_level):
      grayscale_transform_arr[i] = np.round((cumulative_hist[i] - cumulative_hist[g_min]) / (h * w - cumulative_hist[g_min]) * (grayscale_level - 1))
    map_func = lambda x: grayscale_transform_arr[x]
    equalization_img = map_func(hist).astype(np.uint8)

    # build new histogram
    new_fig = plt.figure()
    plt.hist(equalization_img, np.arange(grayscale_level + 1, dtype=int), facecolor='g', alpha=0.75)
    plt.xlabel('GrayScale')
    plt.ylabel('Quantity')
    plt.title('Histogram of GrayScale')
    plt.grid(True)

    # reszie new histogram image and cancatenate images vertically
    new_hist_img = ImageHistogram.array2image(ImageHistogram.figure2array(new_fig))
    equalization_img = Image.fromarray(equalization_img.reshape(h,w), mode='L')
    new_hist_img = self.resize(image, new_hist_img)

    # concatenate histogram and image together
    img_org = self.concat_image_vertical(image, original_hist_img)
    img_new = self.concat_image_vertical(equalization_img, new_hist_img)

    # update image in GUI
    photo_org = ImageTk.PhotoImage(image=img_org)
    photo_org.image = img_org
    self.labels[0].rphoto = photo_org


    photo_new = ImageTk.PhotoImage(image=img_new)
    photo_new.image = img_new
    self.labels[1].rphoto = photo_new

    gui.resize_label_image(self.labels, self.window)

  def resize(self, img1, img2):
    '''
    resize img2 and make img2's width is same as img1, but ratio not changed.  
    input:  
      img1, img2: PIL.Image
    output:  
      img2: PIL.Image
    '''
    img1_w, _ = img1.size
    img2_w, img2_h = img2.size
    img2_h = int(np.round(img2_h * img1_w / img2_w))
    img2 = img2.resize((img1_w,img2_h))
    return img2

  def concat_image_vertical(self, img1, img2):
    '''
    concatenate grayscale image vertically.  
    input: 
      img1, img2: PIL.Image, grayscale image  
    output:
      new_img: PIL.Image
    '''
    new_img = Image.new('L', (img1.width, img1.height + img2.height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    return new_img


class SmoothingConvolution:
  ''' Image Smoothing and Convolution '''
  def __init__(self, window, labels):
    self.window = window
    self.labels = labels
    func_btn = tk.Button(self.window, text="Convolution")
    func_btn.place(x=810, y=5)
    func_btn.bind('<Button-1>', self.smoothing_convolution)

  def smoothing_convolution(self, event):
    image = self.labels[0].rphoto.image
    if not image.mode == 'L':
      image = image.convert('L')

    gau_flt:np.ndarray = self.get_GaussianFlt()
 
    mask:np.ndarray = self.get_mask()

    arr = np.array(image).astype(np.float64)
    output = self.convolution_2D(arr, gau_flt)
    output = self.convolution_2D(arr, mask)
    # normalization
    output = (output * 255. -  output.min()) / (output.max() - output.min()).astype(np.uint8)

    # update image in GUI
    photo_input = ImageTk.PhotoImage(image=image)
    photo_input.image = image
    self.labels[0].rphoto = photo_input

    img_output = Image.fromarray(output)
    photo_output = ImageTk.PhotoImage(image=img_output)
    photo_output.image = img_output
    self.labels[1].rphoto = photo_output

    gui.resize_label_image(self.labels, self.window)

  def convolution_2D(self, arr, mask):
    '''
    output_size = 1 + (input_size + 2 * padding - kernel_size) / stride, stride = 1,  
    The output matches the input shape(same shape), support mask is not square.
    input:  
      arr: 2-d array, input array.
      mask: 2-d array, convolution mask, it doesn't have to be a square matrix.
    '''
    if arr.dtype == np.float64:
      float_bytes = 8
    elif arr.dtype == np.float32:
      float_bytes = 4

    arr = arr.copy()
    mask_h, mask_w =  mask.shape

    # padding configuration
    padding_h,  padding_w = (mask_h - 1) / 2, (mask_w - 1) / 2
    if padding_h - int(padding_h) != 0:
      # mask's height is even length
      need_padding_h = True
    else:
      need_padding_h = False
    if padding_w - int(padding_w) != 0:
      # mask's width is even length
      need_padding_w = True
    else:
      need_padding_w = False
    padding_h,  padding_w = int(padding_h), int(padding_w)

    arr = np.pad(arr, ((padding_h, padding_h), (padding_w, padding_w)), "edge")
    arr_h, arr_w = arr.shape

    # equivalent to convolution's strides(h,w) is (1, 1), convolution mask is same as mask
    conv_shape = (arr_h - mask_h + 1, arr_w - mask_w + 1, mask_h, mask_w)
    conv_strides = (arr_w * float_bytes, float_bytes, arr_w * float_bytes, float_bytes)
    conv_arr = np.lib.stride_tricks.as_strided(arr, shape=conv_shape, strides=conv_strides)
    output = np.zeros((conv_shape[0], conv_shape[1]))

    for row_id in range(conv_shape[0]):
      for col_id in range(conv_shape[1]):
        output[row_id, col_id] = (conv_arr[row_id, col_id, ...] * mask).sum()

    # convolution with even side of mask, corresponding side needs to pad to keep same shape
    if need_padding_h:
      # padding at bottom border
      output = np.pad(output, ((0,1),(0,0)), "edge")
    if need_padding_w:
      # padding at right border
      output = np.pad(output, ((0,0),(0,1)), "edge")
    return output

  def get_mask(self):
    '''
    Get mask until user enter correct string format to convert string to matrix.
    output:
      mask: 2-d array, convolution mask.
    '''
    mask = None
    while not isinstance(mask, np.ndarray):
      mask_str = askstring("mask coefficients", "Input your mask coefficients row by row")
      mask = self.parse_mask_coefficients(mask_str)
    return mask

  def get_known_operator(self, op_name, pattern):
    '''
    get known op's coefficients directly,  
    support sobel(1~8),  kirsch(1~8), prewitt(1~8), laplace(4,8),  
    if not support op, it will return False.  
    input:
      op_name: str, op name.
      pattern: str, digit(1~8)
    output:
      False or array
    '''
    pattern = int(pattern)
    func_dict = {1: lambda x: x, 2: lambda x: np.rot90(x), 3: lambda x: x.T, 4: lambda x: np.rot90(x).T}
    if op_name.lower() == "sobel":
      coefficients1 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
      coefficients2 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    elif op_name.lower() == "kirsch":
      coefficients1 = np.array([[3,3,3],[3,0,3],[-5,-5,-5]])
      coefficients2 = np.array([[3,3,3],[-5,0,3],[-5,3,3]])
    elif op_name.lower() == "prewitt":
      coefficients1 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
      coefficients2 = np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
    elif op_name.lower() == "laplace":
      # 4-neighborhoods
      if pattern == 4:
        return np.array([[0,1,0],[1,-4,1],[0,1,0]])
      # 8-neighborhoods
      elif pattern == 8:
        return np.array([[1,1,1],[1,-8,1],[1,1,1]])
      else:
        # laplace op not support this pattern
        return False
    else:
      # not support this op
      return False

    if pattern in [1,3,5,7]:
      # transform [1,3,5,7] to [1,2,3,4]
      pattern = (pattern + 1) // 2
      return func_dict.get(pattern)(coefficients1)
    elif pattern in [2,4,6,8]:
      # transform [2,4,6,8] to [1,2,3,4]
      pattern = pattern // 2
      return func_dict.get(pattern)(coefficients2)
    else:
      # this op not support this pattern
      return False

  def parse_mask_coefficients(self, mask_str):
    '''
    Use regular expression to parse string to array, support random size mask.  
    If mask_str meets format, function will return array,
    or return Fasle.  
    1 1 1 0 0 0 1 1 1 -> square matrix format, if cannot convert to square, return False,  
    1 1 1 0 0 0 1 1 1 (3,3) -> random size matrix format, (3,3) means matrix shape,    
    or 1 2 1 -1 2 -1 (3,2) or 1 2 1 -1 2 -1 (2,3) , prouct of shape equal number of element or return False,  
    sobel-1, ... ,sobel-8 -> known operator pattern, see `get_known_operator` annotation.  
    input:
      mask_str: str
    output:
     False or array
    '''
    import re
    known_pattern = re.search(r"(?P<name>\w+)-(?P<pattern>\d)", mask_str)
    if known_pattern:
      # known op
      op_name = known_pattern.group("name")
      pattern = known_pattern.group("pattern")
      coefficients = self.get_known_operator(op_name, pattern)
    else:
      mat_shape_pattern = re.search(r" \((?P<shape>\d+,\d+)\)$", mask_str)
      if mat_shape_pattern:
        # not square matrix
        shape:str = mat_shape_pattern.group("shape")
        shape = tuple(map(int, shape.split(',')))
        mask_str = mask_str[:mat_shape_pattern.start()]
        coefficients:list = re.findall(r"-?\d\.?\d*", mask_str)
        if shape[0] * shape[1] != len(coefficients):
          # cannot reshape matrix
          return False
        coefficients = np.fromiter(map(float, coefficients), float).reshape(shape)
      else:
        # square matrix
        coefficients:list = re.findall(r"-?\d\.?\d*", mask_str)
        if (len(coefficients) ** 0.5 - int(len(coefficients) ** 0.5)) or not coefficients:
          # if list length's square root is not integer, it means can not convert to square matrix,
          # or list length is zero
          return False
        side_len = int(len(coefficients) ** 0.5)
        coefficients = np.fromiter(map(float, coefficients), float).reshape((side_len, side_len))

    return coefficients

  def get_GaussianFlt(self):
    '''
    Get Gaussian Filter with size and deviation which is provided by user.
    '''
    dev = askfloat("Set up Gaussian filter deviation", "Gaussian filter deviation: ([0.1, 50])", minvalue=0.1, maxvalue=50., initialvalue=1.)
    side_len = None
    while not side_len:
      side_len = askinteger("Set up Gaussian filter size", "Gaussian filter size: ([3,5,...,25])", minvalue=3, maxvalue=25, initialvalue=3)
      if side_len not in [i for i in range(1,27,2)]:
        side_len = None
    # gaussian filter, x-positive is rightward, y-positive is downward.
    s = np.linspace(-(side_len - 1)//2, (side_len - 1)//2, side_len)
    xv, yv = np.meshgrid(s, s)
    mask = np.exp(-(xv ** 2 + yv ** 2) / (2 * dev ** 2))
    return mask


if __name__ == '__main__':
  g = gui()