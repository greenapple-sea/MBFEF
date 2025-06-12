from autoanchor import kmean_anchors
anchors = kmean_anchors('coco128.yaml', 9, 640, 5.0, 1000, True)