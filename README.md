# Face_landmark_detection
### Prerequisites
Python 3.6+ version should be installed in your system.

### How to run
Download the dlib’s pre-trained facial landmark detector model from (here)[http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2] and extract it in the root of this repository.

Execute the following commands.

```
  pip install requirements.txt

  python detect_landmarks.py
  ```
  ### Test a new image
  Change the image path from main function.
  
  line 30  img_path = <input image path>

output file is saved as Output_faces.jpg int he root directory.
