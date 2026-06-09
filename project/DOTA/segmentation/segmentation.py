import cv2
import numpy as np

def grabcut_from_obb(img_rgb, poly, margin=0.25, iters=5, restrict_to_obb=True):
    """
    img_rgb: RGB-Bild, shape (H, W, 3)
    poly: DOTA-OBB, ann["poly"]
    restrict_to_obb: if true, then the ifnal Mask is not allowed to be outside the obb

       Das müssen wir Grabcut mitgeben:
        cv.GC_BGD     sicherer Hintergrund
        cv.GC_FGD     sicherer Vordergrund
        cv.GC_PR_BGD  wahrscheinlicher Hintergrund
        cv.GC_PR_FGD  wahrscheinlicher Vordergrund
    """
    # Get Hight + Width from the RGB Picture (1024, 1024, 3) = h:1024, w:1024
    h, w = img_rgb.shape[:2]

    #Dota obb poly to np array
    poly = np.array(poly, dtype=np.float32)

    # Axis-aligned Crop around the rotated bounding box
    # This Basically makes a normal rectangle around the BB (none rotated)
    # x = left top x, y = left top y, bw = width, bh = height
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))

    # pad = the bounding box + some of the environment around it to get Background examples for GrabCut
    pad = int(max(bw, bh) * margin) + 4

    # calculating the Crop border for the rectangle
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    crop = img_rgb[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]

    # The OBB are part of the whole picture,  calculate the point inside the crop
    local_poly = poly - np.array([x1, y1], dtype=np.float32)
    local_poly_i = local_poly.astype(np.int32)

    # OBB empty mask for the crop (np array with 0)
    obb_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    # rotated bounding box within the mask are set to one
    # so this means obb_mask == 0 -> Pixel outside of Dota OBB
    # obb_mask == 1 -> Pixel inside of the Dota OBB
    cv2.fillPoly(obb_mask, [local_poly_i], 1)



    # GrabCut-Mask init. GC_PR_BGD = pribably background
    gc_mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)

    # inside of the OBB =Probably foreground
    gc_mask[obb_mask == 1] = cv2.GC_PR_FGD

    # Build a distance map inside the OBB
    # Pixels close to the OBB border have small distance values
    dist = cv2.distanceTransform(obb_mask, cv2.DIST_L2, 3)

    if dist.max() > 0:
        # Mark a very small inner border area as "probably background".
        # Should help GrabCut avoiding including background near the object edges
        # At the start alot of background was catched as well
        inner_border = (dist > 0) & (dist <= 1.5)
        gc_mask[inner_border] = cv2.GC_PR_BGD

    # Create a smaller "sure foreground" area inside the OBB
    # "erode" the OBB so only the center area is treated as safe object
    kernel = np.ones((3, 3), np.uint8)

    # Erode the OBB mask to get a safe foreground region.
    sure_fg = cv2.erode(obb_mask, kernel, iterations=1)

    # Fallback for very small objects.
    # If erosion removes too much, use a small circle in the object center.
    if sure_fg.sum() < 10:
        cx, cy = local_poly.mean(axis=0).astype(int)
        r = max(1, int(min(bw, bh) * 0.08))
        cv2.circle(sure_fg, (cx, cy), r, 1, -1)

    # For GrabCut: this inner region is definitely foreground
    gc_mask[sure_fg == 1] = cv2.GC_FGD

    # Small  kernel for background dilation.
    kernel_bg = np.ones((3, 3), np.uint8)

    # Slightly enlarge the OBB.
    # Everything outside this enlarged OBB is treated as definite background.
    dilated_obb = cv2.dilate(obb_mask, kernel_bg, iterations=1)
    gc_mask[dilated_obb == 0] = cv2.GC_BGD

    # GrabCut internal models for background and foreground.
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # OpenCV GrabCut expects BGR images, not RGB.
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    # Run GrabCut using the initialized mask.
    cv2.grabCut(
        crop_bgr,
        gc_mask,
        None,
        bgd_model,
        fgd_model,
        iters,
        cv2.GC_INIT_WITH_MASK
    )

    # Convert GrabCut result to a binary mask.
    # Foreground and probable foreground become 1, everything else becomes 0.
    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype(np.uint8)

    # Optional: prevent the final mask from going outside the original OBB
    if restrict_to_obb:
        mask = mask * obb_mask

    # Small fixed kernel for closing.
    kernel_close = np.ones((3, 3), np.uint8)

    # Close small holes or missing parts inside the object
    # MORPH_CLOSE fills small gaps without removing thin object parts
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Apply the OBB restriction again after post-processing
    if restrict_to_obb:
        mask = mask * obb_mask

    #  Find all connected foreground regions in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Label 0 is the background
    # If there is more than one foreground region, keep only the largest one
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        mask = (labels == largest_label).astype(np.uint8)

    # Put the crop-sized mask back into a full-size image mask.
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    return full_mask