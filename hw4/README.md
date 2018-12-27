# HW4 Stereo Matching
## Task
* Part 1: Depth from Disparity (in report)
* Part 2: Disparity Estimation

    
## Part 2: Disparity Estimation 
* Usage
    * Simply execute `main.py` to generate images.
        ```
            python3 main.py 
        ```
    * (Optional) Some arguments for bilateral filter.
        ```
            python3 main.py -s 1.0 -r 0.1
        ```
    * (Optional) Generate image throughout the process.
        ```
            python3 main.py -v
        ```
        Check the result in the generated file called `result_final`
        
    * Using `eval_stereo.py` to evaluate bad pixel ratio.
