# HW1 Advanced Color-to-Gray Conversion

## Task
* Conventional Color Conversion (RGB->YUV)
* Advanced Color Conversion

## Conventional Color Conversion
* Apply simple conversion (ref: [wiki](https://en.wikipedia.org/wiki/YUV))
* Usage
     ```
        python3 color2gray.py --mode c 
                              [-p for plot] 
                              [-i (input image)] 
                              [-o (output directory)]
     ```

## Advanced Color Conversion
* Apply **Joint Bilateral Filter** as similarity measurement.
* To justify which parameters fit the most (i.e. minimize L1 distance).
* Refer to: [***Decolorization: is rgb2gray() out?, SIGGRAPH Asia 2013 Technical Briefs***](https://ybsong00.github.io/siga13tb/siga13tb_final.pdf):
* Usage
    ```
        python3 color2gray.py --mode a 
                              [-p for plot] 
                              [-i (input image)] 
                              [-o (output directory)]
    ```
###Results 
   
* Please refer to the directory [advanced](https://github.com/fanoping/Computer-Vision/tree/master/hw1/advanced), files are named by chosen sigma_r and sigma_s.
