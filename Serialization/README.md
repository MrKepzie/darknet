Autocam serialization library
-----------------------------


This library implements reading and writing results of multiple object detections accross an image sequence or video.
The file format is based on [YAML](http://yaml.org). This library depends on the yaml-cpp library (included in the source distribution) to handle the actual parsing of the files.

File Format Specification
--------------------------

The file starts with a line with the following content:

    # Detections

If the first line of the file is not identical to this header, the file reading will fail.

The file contains a map of frame numbers mapped to a list detection. 

Each frame in the list is labeled with a unique number representing the frame number of an image in a sequence. 
A frame may contain one or multiple detection results and it may vary over time.

A detection is defined with a set of parameters:

    - A rectangle indicating the detected area. The rectangle layout is (x1, y1, x2, y2) where x1 < x2 and y1 < y2. x1 being the left edge of the rectangle, y1 the bottom edge, x2 the right edge, y2 the top edge.
        The coordinates of the edges of the rectangles are NOT normalized with respect to the image size.
        If (w,h) is the image size in pixels:
        - (0,0) is the center of the bottom-left pixel
        - (w-1,h-1) is the center of the top-right pixel
        The rectangle is identified in the detection by the token "Rect".

    - A detection score, normalized in the range 0 <= r <= 1, indicating the confidence of the detection. 1 being a ground-truth detection.
        The score is identified in the detection by the token "Score".

    - An object label uniquely identifying the object type, e.g: "person", "billy", "dog"...
      The object label is identified in the detection by the token "Label". Note that one object type may appear in multiple detections in a single frame.

    - An optional UI label: Its purpose is to replace the identifier when displayed on a user interface. If the label is empty, a user interface should use the identifier instead.
        The object label is identified in the detection by the token "UILabel".

File Example:
-------------

    # Detections
    1:
    - {Rect: [295.986, 537.526, 339.486, 581.026], Score: 1, Label: biff}
    - {Rect: [0, 0, 0, 0], Score: 1, Label: linda}
    - {Rect: [0, 0, 0, 0], Score: 1, Label: happy}
    - {Rect: [1040.174, 518.105, 1103.167, 581.097], Score: 1, Label: willy}
    2:
    - {Rect: [296.362, 538.314, 339.862, 581.814], Score: 1, Label: biff}
    - {Rect: [0, 0, 0, 0], Score: 1, Label: linda}
    - {Rect: [0, 0, 0, 0], Score: 1, Label: happy}
    - {Rect: [1038.869, 518.324, 1101.861, 581.3150000000001], Score: 1, Label: willy}
