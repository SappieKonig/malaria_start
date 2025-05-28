import math
import numpy as np
import pandas as pd

def evaluate_lacuna_malaria_metric(
    reference_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    min_overlap: float = 0.5
):
    """
    reference_df: DataFrame with columns
        ['Image_ID', 'class', 'ymin', 'xmin', 'ymax', 'xmax']
    submission_df: DataFrame with columns
        ['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax']
    Returns:
        mAP: float
    """

    def log_average_miss_rate(prec, rec, num_images):
        # if there were no detections of that class
        if prec.size == 0:
            return 0.0, 1.0, 0.0

        fppi    = 1 - prec
        mr      = 1 - rec
        fppi_tmp= np.insert(fppi, 0, -1.0)
        mr_tmp  = np.insert(mr,   0,  1.0)
        ref     = np.logspace(-2.0, 0.0, num=9)
        for i, ref_i in enumerate(ref):
            # find last fppi_tmp ≤ ref_i
            j        = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i]   = mr_tmp[j]
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
        return lamr, mr, fppi

    def voc_ap(rec, prec):
        # make copies and add sentinel values
        rec = rec[:]
        prec = prec[:]
        rec.insert(0, 0.0)
        rec.append(1.0)
        mrec = rec[:]
        prec.insert(0, 0.0)
        prec.append(0.0)
        mpre = prec[:]

        # monotonic precision
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        # area under curve
        i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i-1]]
        ap = 0.0
        for i in i_list:
            ap += (mrec[i] - mrec[i-1]) * mpre[i]
        return ap, mrec, mpre

    # --- Build ground-truth counters and per-image GT lists ---
    gt_counter_per_class      = {}
    counter_images_per_class  = {}
    # group by image
    gt_group = reference_df.groupby("Image_ID")
    gt_data  = {}
    for img_id, df in gt_group:
        gt_data[img_id] = []
        seen_cls = set()
        for _, row in df.iterrows():
            cls = str(row["class"])
            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            gt_data[img_id].append({"class_name": cls, "bbox": bbox, "used": False})
            gt_counter_per_class[cls] = gt_counter_per_class.get(cls, 0) + 1
            if cls not in seen_cls:
                counter_images_per_class[cls] = counter_images_per_class.get(cls, 0) + 1
                seen_cls.add(cls)

    # --- Build prediction lists per image ---
    pred_group = submission_df.groupby("Image_ID")
    pred_data  = {}
    for img_id, df in pred_group:
        pred_data[img_id] = []
        for _, row in df.iterrows():
            pred_data[img_id].append({
                "class_name": str(row["class"]),
                "confidence": float(row["confidence"]),
                "bbox":       [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            })

    classes     = sorted(gt_counter_per_class.keys())
    n_classes   = len(classes)
    sum_AP      = 0.0
    ap_per_class   = {}
    lamr_per_class = {}

    # --- Evaluate per class ---
    for cls in classes:
        # collect all detections of this class
        detections = []
        for img_id, dets in pred_data.items():
            for d in dets:
                if d["class_name"] == cls:
                    detections.append({
                        "file_id":   img_id,
                        "confidence": d["confidence"],
                        "bbox":       d["bbox"]
                    })
        # sort by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        nd = len(detections)
        tp = [0] * nd
        fp = [0] * nd

        # match detections to GT
        for idx, det in enumerate(detections):
            img_id = det["file_id"]
            bb     = det["bbox"]
            ovmax  = -1.0
            gt_match = None

            # find best IoU over GT boxes of same class
            for obj in gt_data.get(img_id, []):
                if obj["class_name"] == cls:
                    bbgt = obj["bbox"]
                    bi = [
                        max(bb[0], bbgt[0]),
                        max(bb[1], bbgt[1]),
                        min(bb[2], bbgt[2]),
                        min(bb[3], bbgt[3]),
                    ]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = (
                            (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                            + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                            - iw * ih
                        )
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax  = ov
                            gt_match = obj

            if ovmax >= min_overlap and gt_match is not None and not gt_match["used"]:
                tp[idx] = 1
                gt_match["used"] = True
            else:
                fp[idx] = 1

        # cumulative sums
        for i in range(1, nd):
            fp[i] += fp[i-1]
            tp[i] += tp[i-1]

        # precision / recall curves
        rec  = [tp[i] / gt_counter_per_class[cls] for i in range(nd)]
        prec = [
            tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
            for i in range(nd)
        ]

        # AP & LAMR
        ap, _, _    = voc_ap(rec[:], prec[:])
        lamr, _, _  = log_average_miss_rate(
            np.array(prec), np.array(rec), counter_images_per_class[cls]
        )

        ap_per_class[cls]   = ap
        lamr_per_class[cls] = lamr
        sum_AP += ap

    mAP = sum_AP / n_classes if n_classes > 0 else 0.0
    return mAP


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    print(evaluate_lacuna_malaria_metric(df, df))
