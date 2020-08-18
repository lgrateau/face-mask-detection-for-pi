# face-mask-detection-for-pi

# Installation
  1.  Clone this code
> ```console
> $ git clone https://github.com/lgrateau/face-mask-detection-for-pi
> $ cd face-mask-detection-for-pi
> ```

   2. Go to the cloned directory
> ```console
> $ cd face-mask-detection-for-pi
> ```   

   3.  Create a virtual environment
> ```console
> $ python3 -m venv ./
> $ source bin/activate
> ```
    4. a Install python package on Mac OSx
> ```console
> $ pip3 install -r requirements.txt
> ```

    4. b Install python package on Windows
> ```console
> $ pip3 install -r requirements.txt
> ```

    4. c Install python package on Rapsberry 
> ```console
> $ pip3 install -r requirements_for_pi.txt
> ```

# Use the mask detector

## Use the virtual env

> ```console
> $ cd face-mask-detection-for-pi
> $ source bin/activate
> ```
 
## Run the mask detector server
> ```console
> $ python3 detect_mask_video.py
> ```

## Open you browser
Open a browser to this url : http://localhost:5000