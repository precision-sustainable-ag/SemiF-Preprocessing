from utils.datasets import BBox

EPS = 1e-16


def bb_iou(boxA: BBox, boxB: BBox):
    """Function to calculate the IoU of two bounding boxes

    Args:
        _boxA (BBox): First bounding box
        _boxB (BBox): Secong bounding box

    Returns:
        _type_: _description_
    """

    _boxA = [
        boxA.top_left[0], -boxA.top_left[1], boxA.bottom_right[0],
        -boxA.bottom_right[1]
    ]
    _boxB = [
        boxB.top_left[0], -boxB.top_left[1], boxB.bottom_right[0],
        -boxB.bottom_right[1]
    ]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(_boxA[0], _boxB[0])
    yA = max(_boxA[1], _boxB[1])
    xB = min(_boxA[2], _boxB[2])
    yB = min(_boxA[3], _boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((_boxA[2] - _boxA[0]) * (_boxA[3] - _boxA[1]))
    boxBArea = abs((_boxB[2] - _boxB[0]) * (_boxB[3] - _boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def generate_hash(box: BBox, auxilliary_hash=None):
    """Generate a unique ID for the BBox. For now, the hash is the BBox.id,
       since the id is assumed to be unique (composed from the image_id).

    Returns:
        str: The unique ID for the box
    """

    box_hash = box.bbox_id
    # If another has is given, incorporate that
    # This is handy if the hash for a pair of bounding boxes is to be
    # determined
    if auxilliary_hash is not None:
        box_hash = ",".join(sorted([auxilliary_hash, box_hash]))
    return box_hash
