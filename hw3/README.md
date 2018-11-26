# HW3 Projective Geometry
## Task
* Part 1: Estimate Homography
* Part 2: Unwarp the QR code on the screen 
* Part 3: Estimate the 3D illusion
* Bonus: Simple Augmented Reality

## Usage
Simply execute `main.py` to generate image for each part.
    
   ```
        python3 main.py
   ```
    
## Part 1: Estimate Homography
* Simple apply **Homography Matrix** for projection
* Reference (Wikipedia): [Homgraphy](https://en.wikipedia.org/wiki/Homography_(computer_vision))

|    img1   |    img2   |   img3    |    img4   |   img5    |
| :-------: | :-------: | :-------: | :-------: | :-------: |
|![wu](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/wu.jpg)|![ding](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/ding.jpg)|![yao](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/yao.jpg)|![kp](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/kp.jpg)|![lee](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/lee.jpg)|![wu](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/wu.jpg)|

|  Source |   Result  |
| ------- | :-------: |
|![Time Square](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/times_square.jpg)|![Result](https://github.com/fanoping/Computer-Vision/blob/master/hw3/part1.png)|

## Part 2: Unwarp the QR code on the screen
* Apply **Backward Warping** to get pixel information (by interpolation)


|  Source |  Result   |
| ------- | :-------: |
|![Screen](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/screen.jpg)|![Result](https://github.com/fanoping/Computer-Vision/blob/master/hw3/part2.png)| 


## Part 3: Estimate the 3D illusion
* Apply **Backward Warping** to get the top view of the cross walk (pixel-wise interpolation is concerned)
* Discussion concerning the warping result is in the report.

|  Source |  Result   |
| ------- | :-------: |
|![Screen](https://github.com/fanoping/Computer-Vision/blob/master/hw3/input/crosswalk_front.jpg)|![Result](https://github.com/fanoping/Computer-Vision/blob/master/hw3/part3.png)| 

## Bonus: Simple Augmented Reality
* Apply **Feature Detection** and **Projective Geometry**
* Using **SIFT** as feature points detector
* Calculate **Homography Matrix**
* Usage
    ```
        python3 simple_ar.py -v [video]
                             -m [marker]
                             -i [input image]
                             -o [output video]
                             -d [for detect marker position]
    ```
* Example Usage
    * Output an projected image in the video (**Default**)
        ```
            python3 simple_ar.py -v ./input/ar_marker.mp4
                                 -m ./input/marker.png
                                 -i ./input/kp.jpg
                                 -o output_kp.mp4
        ```
        Example output: `output_kp.mp4`
        
    * Output the detected position in the video
        ```
            python3 simple_ar.py -v ./input/ar_marker.mp4
                                 -m ./input/marker.png
                                 -o detectlines.mp4
                                 -d
        ```
        Example output: `detectlines.mp4`
        
