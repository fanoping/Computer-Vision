# Final Project: Depth Generation for More Realistic Scene

## Usage
   * Data is provided in `data` file
   * Execute `main.py` to generate depth images
   * Ex:
        ```
            python3 main.py --input-left [left_image.png] --input-right [right_image.png] --output [output.pfm]
        ```
   * Extra parameters
        
        `-v` for visualize process
        
        `-s` for bilateral filter spatial kernel standard deviation, default = 2.0
        
        `-r` for bilateral filter range kernel standard deviation, default = 0.1
        
## Visualization and Utils
   * Please refer to the file `util.py` and `visualization.py`
   * Read or write **.pfm** file functions are in `util.py`  
